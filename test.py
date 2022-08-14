import argparse
import torch
import numpy as np
from scipy.spatial import distance
import pandas as pd

# my files
from embedding_translator import load_checkpoint
from embedder_train import get_dataloader
from data_preprocess import filname2testset, filname2patentset
from common_utils import set_seed, input2output, get_random_list, input2all, generate_results_file, \
    valid_results_file_to_metrics, process_results_file
from property_handler import rdkit_no_error_print, property_calc, similarity_calc, is_valid_molecule
from embedder import load_checkpoint_embedder
from embedder_train import input2output_embedder

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Unpaired Generative Molecule-to-Molecule Translator'
    )
    # end-end model settings
    parser.add_argument('--check_testset', default=True, action='store_true', help='get test results')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for end-end model')
    parser.add_argument('--enhanced_property', type=str, default='QED', help='name of property to translate (should be folder with that name inside dataset)')
    parser.add_argument('--suppressed_property', type=str, default='patentability', help='patentability / Toxicity')
    parser.add_argument('--testset_filename', type=str, default='A_test.txt', help='testset filename')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints', help='name of folder for checkpoints loading')
    parser.add_argument('--plots_folder', type=str, default='plots_output', help='name of folder for plots saving')
    parser.add_argument('--num_retries', type=int, default=20, help='number of retries for each test sample (K)')
    parser.add_argument('--SR_similarity', type=int, default=0.3, help='minimal similarity for success')
    parser.add_argument('--SR_patentability', type=int, default=0.2, help='maximal distance from patents for success')
    parser.add_argument('--SR_property_QED', type=int, default=0.8, help='minimal QED property value for success')
    parser.add_argument('--seed', type=int, default=50, help='base seed')
    parser.add_argument('--test_max_len', type=int, default=90, help='length of test smiles')
    parser.add_argument('--is_CDN', default=False, action='store_false', help='test CDN network')
    parser.add_argument('--patentset_path', type=str, default='dataset/QED/SureChEMBL.txt', help='the patents dataset path')

    # ablation
    parser.add_argument('--use_fp', default=True, action='store_true', help='does translator use molecule fp')
    parser.add_argument('--use_baseline_1', default=False, action='store_true', help='Only one cycle experiment')
    parser.add_argument('--use_baseline_2', default=False, action='store_true', help='Two separate cycles experiment')
    parser.add_argument('--use_baseline_3', default=False, action='store_true', help='Add EET-Net component experiment')
    parser.add_argument('--no_EET_Net', default=False, action='store_true', help='disable EET-Net')
    parser.add_argument('--no_DEET_Net', default=False, action='store_true', help='disable DEET-Net')

    args = parser.parse_args()
    return args


# check testset
def check_testset(args, model_in, model_out, input2output_func):
    testset_path = 'dataset/' + args.enhanced_property + '/' + args.testset_filename
    results_file_path = args.plots_folder + '/' + args.enhanced_property + '/MOMNEC_' + args.enhanced_property + '_test.txt'
    valid_results_file_path = args.plots_folder + '/' + args.enhanced_property + '/valid_MOMNEC_' + args.enhanced_property + '_test.txt'
    print(' ')
    print('Loading testset from file   => ' + testset_path)

    # train set for novelty
    trainset = set(model_in.dataset.trainset).union(model_out.dataset.trainset)

    # build testset from filename
    testset_list = filname2testset(testset_path, model_in, model_out)
    test_loader = get_dataloader(model_in, args, testset_list, batch_size=args.batch_size, shuffle=False)

    # build patentset from filename
    print(' ')
    print('Loading patentset from file   => ' + args.patentset_path)

    # load patentset
    if args.suppressed_property is 'patentability':
        patentset = filname2patentset(args.patentset_path)
    else:
        patentset = None

    # generate random seeds
    random_seed_list = get_random_list(args.num_retries)

    def input2output_func_aux(input_batch):
        return input2output_func(input_batch, random_seed_list)

    def input2smiles(input):
        return model_in.tensor2string(input)  # to smiles

    # generate results file
    generate_results_file(test_loader, input2output_func_aux, input2smiles, results_file_path)

    # result file -> valid results + property and similarity for output molecules
    process_results_file(results_file_path, args, valid_results_file_path, trainset, patentset)

    testset_len = 800

    # calculate metrics
    validity_mean, validity_std, \
    diversity_mean, diversity_std, \
    novelty_mean, novelty_std, \
    property_mean, property_std, \
    similarity_mean, similarity_std, \
    SR_mean, SR_std, \
    PR_mean, PR_std = \
        valid_results_file_to_metrics(valid_results_file_path, args, testset_len)

    # print results
    print(' ')
    print('Property => ' + args.enhanced_property)
    print('property => mean: ' + str(round(property_mean, 3)) + '   std: ' + str(round(property_std, 3)))
    print(args.suppressed_property + ' => mean: ' + str(round(PR_mean, 3)) + '   std: ' + str(round(PR_std, 3)))
    print('fingerprint Similarity => mean: ' + str(round(similarity_mean, 3)) + '   std: ' + str(
        round(similarity_std, 3)))
    print('SR => mean: ' + str(round(SR_mean, 3)) + '   std: ' + str(round(SR_std, 3)))
    print('validity => mean: ' + str(round(validity_mean, 3)) + '   std: ' + str(round(validity_std, 3)))
    print('novelty => mean: ' + str(round(novelty_mean, 3)) + '   std: ' + str(round(novelty_std, 3)))
    print('diversity => mean: ' + str(round(diversity_mean, 3)) + '   std: ' + str(round(diversity_std, 3)))


if __name__ == "__main__":

    with torch.no_grad():
        # parse arguments
        args = parse_arguments()

        # set seed
        set_seed(args.seed)

        # set device (CPU/GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        # disable rdkit error messages
        rdkit_no_error_print()


        # load checkpoint
        T_AB, T_BA, model_A, model_B, T_BC, T_CB, model_C, best_criterion = load_checkpoint(args, device)

        # evaluation mode
        T_AB.eval()
        T_BA.eval()
        model_A.eval()
        model_B.eval()
        T_BC.eval()
        T_CB.eval()
        model_C.eval()

        # initialize property value predictor
        print('Loaded model best criterion validation value: ' + str(best_criterion))

        # prepare models according to translation direction
        model_in, T1, T2, model_out = (model_A, T_AB, T_BC, model_C)

        # check testset
        if args.check_testset is True:
            def test_input2output_func(input_batch, random_seed_list):
                return input2output(args, input_batch, model_in, T1, T2, model_out, random_seed_list, max_out_len=args.test_max_len)
            check_testset(args, model_in, model_out, test_input2output_func)

