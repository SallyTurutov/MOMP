Unpaired Generative Molecule-to-Molecule Translation code.


# Installation:
1. Install conda / minicaonda
2. From the main folder run:\
    i. conda env create -f environment.yml\
    ii. conda activate MOMP

All dataset files located in dataset folder.

# Training:
From the main folder run:

    python train.py 2>error_train.txt

train.py is the main training file, contains most of the hyper-parameter and configuration setting.
After training, the checkpoints will be located in the checkpoints folder.

Main setting:\
enhanced_property (line 25) -> QED. \
suppressed_property (line 26) -> patentability (= PL).


# Inference  :
From the main folder run:

    python test.py 2>error_test.txt

test.py is the main testing file, contains most of the hyper-parameter and configuration setting.

Main setting:\
check_testset (line 23) [Main Results: Molecule Optimization] -> True / False.


# Ablation Experiments  :
Druing training, run regularly except for these changes:
1. No Pre-train		    -> no_pre_train_models to True.
2. NO Dist-loss 		-> no_dist_loss to True.
3. NO fp 			    -> use_fp to False.
4. Two separate cycles  -> use_two_separate_cycles to True.
5. Add EETN 	        -> use_add_EETN to True.

For testing, run regularly except for these changes:
1. NO fp 			    -> use_fp to False.
2. Two separate cycles  -> use_two_separate_cycles to True.
3. Add EETN 	        -> use_add_EETN to True.
