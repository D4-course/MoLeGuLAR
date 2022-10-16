import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

st.title("MoLeGuLAR")

st.header("Select Pretrained Model")

kwargs = {
    'reward': 'exponential',
    'calculator': 'docking',
    'receptor': '6LU7',
    'LogP_target': 2.5,
    'QED_target': None,
    'hydration_target': None,
    'TPSA_target': None,
    'strategy': 'switch'
}

st.subheader("Reward Type")
kwargs["reward"] = st.selectbox(
    "Reward",
    ["exponential"],
    index=0)
kwargs["strategy"] = st.selectbox(
    "Strategy",
    ["switch", "sum"],
    index=0
)

st.subheader("Binding Affinity")
kwargs["calculator"] = st.selectbox(
    "Calculator",
    ["docking", "gin"],
    index=0
)
kwargs["receptor"] = st.selectbox(
    "Receptor",
    ["4BTK", "6LU7"],
    index=1
)

st.subheader("Molecular Properties")
kwargs["LogP_target"] = st.selectbox(
    "LogP",
    [None, "2.5"],
    index=1
)
kwargs["QED_target"] = st.selectbox(
    "QED",
    [None, "1"],
    index=0
)
kwargs["hydration_target"] = st.selectbox(
    "Solvation",
    [None, "7", "11"],
    index=0
)
kwargs["TPSA_target"] = st.selectbox(
    "TPSA",
    [None, "70", "120"],
    index=0
)

temp_data = {
    "Smile": [
        "C=C1C(c2ccccc2OC)=NC(=O)C1c1ccccc1OC(C)=O",
        "CC(C)CN1C(=O)N2c3ccccc3OCC2C(=O)C=C1Sc1ccccc1",
        "CCN(CC)C(=O)CCN1C(=O)c2ccccc2-c2cnccc21",
        "CCN1CCC(C(N)CC(=O)N2CCN(CCC(=O)NCC(=O)NC)CC2)CC1"
        ],
    "Binding Affinity": [
        -8.28,
        -10.56,
        -12.37,
        -7.90
    ],
    "LogP": [
        3.2,
        4.3,
        6.9,
        2.7
    ],
    "QED": [
        0.11,
        0.64,
        0.32,
        0.97
    ],
    "TPSA": [
        64.76,
        141.24,
        49.85,
        71.11
    ],
    "Solvation": [
        -9.8,
        -10.9,
        -8.5,
        -6.7
    ]
}

st.subheader("Number of Molecules")
NUM_TO_GENERATE = st.slider(
    "Number of Molecules to Generate",
    min_value=1,
    max_value=300,
    value=50,
    step=1
)

# df = pd.DataFrame(temp_data)
# df.to_pickle("my_molecules.pkl")

st.subheader("Results")
is_run_code = st.button("Run Code")
if is_run_code:
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
    import pickle5 as pickle
    import argparse

    from analysis import load_pretrained, estimate_and_update, get_all_props
    
    model, predictor = load_pretrained(**kwargs)
    smiles, predictions = estimate_and_update(
        model, predictor, NUM_TO_GENERATE, kwargs['receptor'])
    df = get_all_props(smiles, predictions)
    df.to_pickle("my_molecules.pkl")


st.subheader("Data")
df = pd.read_pickle("my_molecules.pkl")
st.dataframe(df)

st.subheader("Plots")
col = st.selectbox(
    "Select a column",
    df.columns[1:],
    index=1)
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.kdeplot(df[col], shade=True)
st.pyplot()
