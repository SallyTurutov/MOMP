Unpaired Generative Molecule-to-Molecule Translation code.

We run all training and experiments on Ubuntu 18.04.5 using one Nvidia GeForce RTX 2080 Ti 11GB GPU, two Intel
13 Xeon Gold 6230 2.10GHZ CPUs and 64GB RAM memory.


# Installation:
1. Install conda / minicaonda
2. From the main folder run:\
    i. conda env create -f environment.yml\
    ii. conda activate MOMP

All dataset files located in dataset folder.

# Training:
From the main folder run:

1. python train.py 2>error_train.txt

train.py is the main training file, contains most of the hyper-parameter and configuration setting.
After training, the checkpoints will be located in the checkpoints folder.

Main setting:\
enhanced_property (line 25) -> QED. \
suppressed_property (line 26) -> patentability (= PL).


# Inference  :
From the main folder run:

1. python test.py 2>error_test.txt

test.py is the main testing file, contains most of the hyper-parameter and configuration setting.

Main setting:\
check_testset (line 23) [Main Results: Molecule Optimization] -> True / False.


# Ablation Experiments  :
Druing training::no_EET_Net
1. No Pre-train		    -> no_pre_train_models to True.
2. NO EETN 			    -> no_EET_Net to True.
3. NO fp 			    -> use_fp to False.
4. Two separate cycles  -> use_baseline_2 to True.
5. Add EETN 	        -> use_baseline_3 to True.

For testing, run regularly except for these changes:
1. NO EETN 			-> no_EET_Net to True.
2. NO fp 			-> use_fp to False.
3. Only fp			-> use_fp to False.
