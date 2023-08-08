import torch
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# my files
from common_utils import generate_results_file, valid_results_file_to_metrics, process_results_file


# validation function
def general_validation(args, input2output, input_tensor2string, valid_loader, validset_len, model_name, trainset):
    with torch.no_grad():
        results_file_path = args.plots_folder + '/' + args.property + '/' + args.property + '_validation.txt'
        valid_results_file_path = args.plots_folder + '/' + args.property + '/valid_' + args.property + '_validation.txt'

        # generate results file
        generate_results_file(valid_loader, input2output, input_tensor2string, results_file_path)

        # result file -> valid results + property and similarity for output molecules
        process_results_file(results_file_path, args, valid_results_file_path, trainset)

        # calculate metrics
        validity_mean, validity_std, \
        diversity_mean, diversity_std, \
        novelty_mean, novelty_std, \
        property_mean, property_std, \
        similarity_mean, similarity_std, \
        SR_mean, SR_std, \
        PR_mean, PR_std = \
            valid_results_file_to_metrics(valid_results_file_path, args, validset_len)

        # print results
        print(' ')
        # print('All results are calculated only for patentable molecules!')
        print('Property => ' + args.property)
        print('Model name   => ' + model_name)
        print('property => mean: ' + str(round(property_mean, 3)) + '   std: ' + str(round(property_std, 3)))
        print(args.patentability + ' => mean: ' + str(round(PR_mean, 3)) + '   std: ' + str(round(PR_std, 3)))
        print('fingerprint Similarity => mean: ' + str(round(similarity_mean, 3)) + '   std: ' + str(
            round(similarity_std, 3)))
        print('SR => mean: ' + str(round(SR_mean, 3)) + '   std: ' + str(round(SR_std, 3)))
        print('validity => mean: ' + str(round(validity_mean, 3)) + '   std: ' + str(round(validity_std, 3)))
        print('novelty => mean: ' + str(round(novelty_mean, 3)) + '   std: ' + str(round(novelty_std, 3)))
        print('diversity => mean: ' + str(round(diversity_mean, 3)) + '   std: ' + str(round(diversity_std, 3)))

        return similarity_mean, property_mean, SR_mean, PR_mean, validity_mean, novelty_mean, diversity_mean
