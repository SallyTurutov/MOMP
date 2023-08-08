Molecular Optimization Model with Patentability Constraint (MOMP)

## Installation:
1. Install conda or miniconda. 
2. Set up the environment by running the following commands from the main folder: \
   i. `conda env create -f environment.yml` \
   ii. `conda activate MOMP` 
3. Download the SureChEMBL dataset from: [SureChEMBL Dataset](http://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBLccs/) \
   i. Place the downloaded dataset in the `dataset/SureChEMBL` folder. \
   ii. Run: `python handle_patents_dataset.py` from the main folder. \
   iii. Move the generated `SureChEMBL.txt` file to `dataset/SureChEMBL`. 

**Note:** Alternatively, to save time on preprocessing the data, you can download the `patentsFp.pkl` file from the following link: [patentsFp.pkl](https://drive.google.com/file/d/1Ry2u9kmfEN6donEtUWlT3Dx1W4qho-hJ/view?usp=sharing). Place the downloaded `patentsFp.pkl` file in the `dataset/PL/` folder.

## Training:
From the main folder, execute the following command:

    python train.py 2>error_train.txt

`train.py` is the main training file containing most of the hyper-parameters and configuration settings. After training, the checkpoints will be saved in the `checkpoints` folder.

Main Settings:
- `epochs`: Number of epochs to train the end-to-end model (QED=12 / DRD2=18).
- `property`: Name of the property to optimize - QED / DRD2.
- `SR_similarity`: Minimal similarity required for success (QED=0.15 / DRD2=0.15).
- `SR_patentability`: Maximum distance from patents for success (QED=0.4 / DRD2=0.6).
- `SR_property`: Minimal property value for success (QED=0.7 / DRD2=0.7).
- `use_fp`: Whether the translator uses molecule fingerprints (True/False).
- `use_C_fp`: Whether the translator uses molecule C-fingerprints (True/False).
- `no_pre_train_models`: Disable METNs pre-training (True/False).
- `load_checkpoint`: Load existing checkpoint (True/False).

## Inference:
From the main folder, execute the following command:

    python test.py 2>error_test.txt

## Ablation Experiments:
Results of all the ablation experiments can be found in the `ablation_outputs` folder.

## Baselines:
Results of all the baseline experiments can be found in the `baselines_outputs` folder.

**Note:** The MOMP model can be avoided from training, as the test files for the MOMP model with QED and DRD2 can be found in `baselines_outputs\QED` and `baselines_outputs\DRD2`, respectively.
