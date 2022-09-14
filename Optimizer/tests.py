import os


my_dict = {
    "Single Objective: Binding Affinity with SARS-CoV-2 Mpro using Docking Calculation": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor dock --protein 6LU7",
    "Single Objective: Binding Affinity with TTBK1 using Docking Calculation": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor dock --protein 4BTK",
    "Single Objective: Binding Affinity with TTBK1 using GIN": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor gin --protein 4BTK",
    "Multi Objective : Binding Affinity with SARS-CoV-2 Mpro using Docking Calculation and target LogP = 2.5 (sum)": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor dock --protein 6LU7 --logP yes --logP_threshold 2.5 --switch no",
    "Multi Objective : Binding Affinity with TTBK1 using Docking Calculation and target LogP = 2.5 (sum)": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor dock --protein 4BTK --logP yes --logP_threshold 2.5 --switch no",
    "Multi Objective : Binding Affinity with TTBK1 using GIN and target LogP = 2.5 (sum)": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor gin --protein 4BTK --logP yes --logP_threshold 2.5 --switch no",
    "Multi Objective : Binding Affinity with TTBK1 using GIN, target LogP = 2.5, target QED = 1, target TPSA = 100 A2 and target ΔGHyd = -10 kcal/mol (sum)": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor gin --protein 4BTK --logP yes --logP_threshold 2.5 --qed yes --qed_threshold 1 --solvation yes --solvation_threshold -10 --tpsa yes --tpsa_threshold 100 --switch no",
    "Multi Objective : Binding Affinity with SARS-CoV-2 Mpro using Docking Calculation and target LogP = 2.5 (alternate)": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor dock --protein 6LU7 --logP yes --logP_threshold 2.5 --switch yes",
    "Multi Objective : Binding Affinity with TTBK1 using Docking Calculation and target LogP = 2.5 (alternate)": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor dock --protein 4BTK --logP yes --logP_threshold 2.5 --switch yes",
    "Multi Objective : Binding Affinity with TTBK1 using GIN, target LogP = 2.5 and target QED = 1 (alternate)": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor gin --protein 4BTK --logP yes --logP_threshold 2.5 --switch yes --qed yes --qed_threshold 1",
    "Multi Objective : Binding Affinity with TTBK1 using GIN, target LogP = 6 and target QED = 1 (alternate)": "python model_logP_QED_switch.py --reward_function exponential --num_iterations 175 --predictor gin --protein 4BTK --logP yes --logP_threshold 6 --switch yes --qed yes --qed_threshold 1",
    "Multi Objective : Binding Affinity with TTBK1 using GIN and different targets of TPSA and GHyd": "python model_hydration_tpsa_switch.py --reward_function exponential --num_iterations 175 --predictor gin --protein 4BTK --solvation yes --tpsa yes --solvation_threshold -10 --tpsa_threshold 10 --switch yes"
}

testing = 0
while testing < 10:
    testing += 1
    p1 = input("Would you like to train the model? (y/n): ")
    if p1 == 'y' or p1 == 'Y':
        for i, (k, cmd) in enumerate(my_dict.items()):
            print(f"\nType {i} to perform {k}")
        p3 = input("\nEnter a value [0-11]: ")
        os.system('clear')
        for i, (k, cmd) in enumerate(my_dict.items()):
            if int(p3) == i:
                print(f"\nExecute {k} using:\n\n{cmd}\n")
                os.system(cmd)
    else:
        p2 = input("Would you like to evaluate the model? (y/n): ")
        os.system('clear')
        if p2 == 'y' or p2 == 'Y':
            cmd = "python ../Analysis/analysis.py"
            print(f"Run the following command:\n\n{cmd}\n")
            os.system(cmd)
        else:
            break
