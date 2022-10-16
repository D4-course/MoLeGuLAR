import sys
sys.path.append('../Optimizer/release')
sys.path.append('../Optimizer')

from data import GeneratorData
from tqdm.notebook import tqdm
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from Predictors.SolvationPredictor import FreeSolvPredictor
from rdkit.Chem.QED import qed
from rdkit.Chem import Crippen
from joblib import Parallel, delayed
import os
from utils import canonical_smiles
from stackRNN import StackAugmentedRNN
from rdkit.Chem import AllChem, rdmolfiles
from rdkit import Chem, DataStructs
import pickle
from tqdm import tqdm, trange
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pickle5 as pickle
import pandas as pd
import argparse

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("--reward_function", help="Reward Function linear/exponential/log/squared", default="exponential")
parser.add_argument("--gen_data", default='../Optimizer/random.smi')
parser.add_argument("--num_molecules", default=50, type=int, help="Number of molecules to generate")
parser.add_argument("--strategy", default='switch', help="switch reward function")
parser.add_argument("--calculator", default='docking', help='Choose prediction algorithm')
parser.add_argument("--protein", default='4BTK', help='4BTK/6LU7')
parser.add_argument("--logP", default=2.5, type=float)
parser.add_argument("--tpsa", default=None, type=float)
parser.add_argument("--solvation", default=None, type=float)
parser.add_argument("--qed", default=None, type=float)
args = parser.parse_args()

gen_data_path = args.gen_data

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=tokens)


def dock_and_get_score(smile: str, index: int, receptor: str) -> float:
    """
        smile: SMILES string of the molecule
        index: Index of the molecule
        Receptor: PDB ID of receptor
    """
    try:
        path = "../mgltools_x86_64Linux2_1.5.6/bin/python2.5 ../mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24"
        mol = Chem.MolFromSmiles(smile)
        AllChem.EmbedMolecule(mol)
        if not os.path.exists("molecules"):
            os.mkdir("molecules")
        if not os.path.exists("logs"):
            os.mkdir("logs")
        rdmolfiles.MolToPDBFile(mol, "molecules/" + str(index) + ".pdb")

        os.system(
            f"{path}/prepare_ligand4.py -l molecules/{str(index) + '.pdb'} -o molecules/{str(index) + '.pdbqt'}")
        os.system(f"{path}/prepare_receptor4.py -r ../Optimizer/{receptor}.pdb")
        os.system(
            f"{path}/prepare_gpf4.py -i ../Optimizer/{receptor}_ref.gpf -l molecules/{str(index) + '.pdbqt'} -r {receptor}.pdbqt")

        os.system(f"autogrid4 -p {receptor}.gpf > /dev/null 2>&1")
        os.system(
            f"../AutoDock-GPU/bin/autodock_gpu_128wi -ffile {receptor}.maps.fld -lfile molecules/{str(index) + '.pdbqt'} -resnam logs/{str(index)} -nrun 5 -devnum 1")

        cmd = f"cat logs/{str(index) + '.dlg'} | grep -i ranking | tr -s '\t' ' ' | cut -d ' ' -f 5 | head -n1"
        stream = os.popen(cmd)
        output = float(stream.read().strip())
        return output
    except Exception as e:
        print(f"Did Not Complete because of {e}")
        return 0


class Predictor(object):
    def __init__(self):
        super(Predictor, self).__init__()

    def predict(self, smiles, receptor, use_tqdm=False):
        canonical_indices = []
        invalid_indices = []
        if use_tqdm:
            pbar = tqdm(range(len(smiles)))
        else:
            pbar = range(len(smiles))
        for i in pbar:
            sm = smiles[i]
            if use_tqdm:
                pbar.set_description("Calculating predictions...")
            try:
                sm = Chem.MolToSmiles(Chem.MolFromSmiles(sm))
                if len(sm) == 0:
                    invalid_indices.append(i)
                else:
                    canonical_indices.append(i)
            except:
                invalid_indices.append(i)
        canonical_smiles = [smiles[i] for i in canonical_indices]
        invalid_smiles = [smiles[i] for i in invalid_indices]
        if len(canonical_indices) == 0:
            return canonical_smiles, [], invalid_smiles
        prediction = [dock_and_get_score(
            smiles[index], index, receptor) for index in tqdm(canonical_indices)]
        return canonical_smiles, prediction, invalid_smiles


def plot_hist(prediction, n_to_generate):
    prediction = np.array(prediction)
    ax = sns.kdeplot(prediction, shade=True)
    ax.set(xlabel='Predicted Docking Score',
           title='Distribution of predicted Docking for generated molecules')
    plt.show()


def estimate_and_update(generator, predictor, n_to_generate, receptor):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])

    sanitized = canonical_smiles(
        generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles, receptor)

    plot_hist(prediction, n_to_generate)

    return smiles, prediction


def load_pretrained(
        reward, calculator, receptor,
        LogP_target=None, QED_target=None,
        hydration_target=None, TPSA_target=None, strategy=None):
    """
        reward: Exponential/Linear (All current models use exponential)
        calculator: docking/gin
        receptor: 4BTK/6LU7
        LogP_target: Optional
        QED_target: Optional
        hydration_target: Optional
        TPSA_target: Optional
        Strategy: None for Single Objective and sum/switch for Multiobjective
    """

    file_name = f'../models/{calculator}/model_{reward}_{calculator}_{receptor}'
    if LogP_target:
        file_name += f"_LogP{LogP_target}"
    if QED_target:
        file_name += f"_QED{QED_target}"
    if hydration_target:
        file_name += f"_solvation{hydration_target}"
    if TPSA_target:
        file_name += f"_tpsa{TPSA_target}"
    if strategy:
        file_name += f"_{strategy}"
    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                     output_size=gen_data.n_characters, layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance, lr=lr)
    my_generator.load_model(file_name)

    if calculator == 'docking':
        pred = Predictor()
    elif calculator == 'gin':
        from Predictors import GINPredictor
        pred = GINPredictor.Predictor(
            '../Optimizer/Predictors/GINPredictor.tar')
    return my_generator, pred

kwargs = {
    'reward': args.reward_function,
    'calculator': args.calculator,
    'receptor': args.protein,
    'LogP_target': args.logP,
    'QED_target': args.qed,
    'hydration_target': args.solvation,
    'TPSA_target': args.tpsa,
    'strategy': args.strategy
}
NUM_TO_GENERATE = args.num_molecules

model, predictor = load_pretrained(**kwargs)

smiles, predictions = estimate_and_update(
    model, predictor, NUM_TO_GENERATE, kwargs['receptor'])


def get_all_props(smiles, predictions):
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    LogPs = [Crippen.MolLogP(mol) for mol in mols]
    solv_predictor = FreeSolvPredictor(
        '../Optimizer/Predictors/SolvationPredictor.tar')
    _, hydrations, _ = solv_predictor.predict(smiles, use_tqdm=False)
    tpsas = [CalcTPSA(mol) for mol in mols]
    qeds = []
    for mol in mols:
        try:
            qeds.append(qed(mol))
        except:
            pass
    return pd.DataFrame({
        'Smile': smiles,
        'Binding Affinity': predictions,
        'LogP': LogPs,
        'QED': qeds,
        'TPSA': tpsas,
        '$\Delta G_{Hyd}$': hydrations
    })


df = get_all_props(smiles, predictions)

# plt.figure(figsize=(10, 10))
# plt.rcParams.update({'font.size': 24})
# sns.kdeplot(df['Binding Affinity'], shade=True)
