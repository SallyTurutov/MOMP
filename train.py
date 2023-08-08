import argparse
import torch
import itertools

# my files
from data_preprocess import create_dataset, Dataset, create_dynamic_dataset
from embedder import Embedder
from embedder_train import fit, get_model_train_params, get_dataloader
from embedding_translator import Translator, weights_init_normal, LambdaLR, Statistics, save_checkpoint, load_checkpoint
from property_handler import smiles2fingerprint, rdkit_no_error_print, property_init
from validation import general_validation
from common_utils import set_seed, input2output, get_random_list


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Unpaired Generative Molecule-to-Molecule Translator'
    )

    # end-end model settings
    parser.add_argument('--epochs', type=int, default=18, help='number of epochs to train the end-end model (QED=12 / DRD2=18)')
    parser.add_argument('--epoch_init', type=int, default=1, help='initial epoch')
    parser.add_argument('--epoch_decay', type=int, default=90, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for end-end model')
    parser.add_argument('--property', type=str, default='DRD2', help='name of property to translate - QED / DRD2')
    parser.add_argument('--patentability', type=str, default='PL', help='named PL in the article')
    parser.add_argument('--init_lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--is_valid', default=True, action='store_true', help='run validation every train epoch')
    parser.add_argument('--rebuild_dataset', default=False, action='store_false', help='rebuild dataset files')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints', help='name of folder for checkpoints saving')
    parser.add_argument('--plots_folder', type=str, default='plots_output', help='name of folder for plots saving')
    parser.add_argument('--early_stopping', type=int, default=15, help='Whether to stop training early if there is no\
                        criterion improvement for this number of validation runs.')
    parser.add_argument('--seed', type=int, default=50, help='base seed')
    parser.add_argument('--num_retries', type=int, default=20, help='number of retries for each validation sample')
    parser.add_argument('--patentset_path', type=str, default='dataset/QED/SureChEMBL.txt', help='the patents dataset path')
    parser.add_argument('--SR_similarity', type=int, default=0.15, help='minimal similarity for success (QED=0.15 / DRD2=0.15)')
    parser.add_argument('--SR_patentability', type=int, default=0.6, help='maximal distance from patents for success (QED=0.4 / DRD2=0.6)')
    parser.add_argument('--SR_property', type=int, default=0.7, help='minimal property value for success (QED=0.7 / DRD2=0.7)')
    parser.add_argument('--validation_max_len', type=int, default=90, help='length of validation smiles')
    parser.add_argument('--validation_batch_size', type=int, default=32, help='batch size for validation end-end model')
    parser.add_argument('--validation_freq', type=int, default=3, help='validate every n-th epoch')
    parser.add_argument('--tokenize', default=False, action='store_false', help='use atom tokenization')
    parser.add_argument('--cycle_loss', default=True, action='store_true', help='use cycle loss during training or not')
    parser.add_argument('--EETN_loss_coef', type=int, default=2, help='UGMMT loss coef (loss_AB)')
    parser.add_argument('--Extended_EETN_loss_coef', type=int, default=2, help='UGMMT loss coef (loss_AP)')

    # Ablation
    parser.add_argument('--kl_loss', default=True, action='store_true', help='use kl loss during training or not')
    parser.add_argument('--use_fp', default=True, action='store_true', help='does translator use molecule fp')
    parser.add_argument('--use_C_fp', default=True, action='store_true', help='does translator use molecule C-fp')
    parser.add_argument('--use_baseline_1', default=False, action='store_true', help='Only one cycle experiment')
    parser.add_argument('--no_pre_train_models', default=False, action='store_true', help='disable METNs pre training')
    parser.add_argument('--load_checkpoint', default=False, action='store_true', help='loading existing checkpoint')

    args = parser.parse_args()
    return args


def train_iteration_EETN(real_A, real_B, model_A, model_B, T_AB, T_BA, args):

    real_A, fp_mols_A = real_A
    real_B, fp_mols_B = real_B

    # embedder (METN)
    real_A_emb, kl_loss_A = model_A.forward_encoder(real_A)
    real_B_emb, kl_loss_B = model_B.forward_encoder(real_B)
    if args.kl_loss is False:
        kl_loss_A, kl_loss_B = None, None

    # prepare fingerprints
    if args.use_fp:  # regular
        if args.use_C_fp:
            real_A_fp_str = [smiles2fingerprint(model_A.tensor2string(mol), fp_translator=True) for mol in fp_mols_A]
            real_B_fp_str = [smiles2fingerprint(model_B.tensor2string(mol), fp_translator=True) for mol in fp_mols_B]
        else:
            real_A_fp_str = [smiles2fingerprint(model_A.tensor2string(mol), fp_translator=True) for mol in real_A]
            real_B_fp_str = [smiles2fingerprint(model_B.tensor2string(mol), fp_translator=True) for mol in real_B]
        real_A_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_A_fp_str]).to(device)
        real_B_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_B_fp_str]).to(device)
        real_A_fp = real_A_fp.detach()
        real_B_fp = real_B_fp.detach()
    else:  # ablation
        real_A_fp, real_B_fp = None, None

    fake_B_emb = T_AB(real_A_emb, real_A_fp)
    fake_A_emb = T_BA(real_B_emb, real_B_fp)

    # Cycle loss
    cycle_A_emb = T_BA(fake_B_emb, real_A_fp)
    cycle_B_emb = T_AB(fake_A_emb, real_B_fp)

    cycle_loss_A, _, recon_A = model_A.forward_decoder(real_A, cycle_A_emb, get_recon_strings=True)
    cycle_loss_B, _, recon_B = model_B.forward_decoder(real_B, cycle_B_emb, get_recon_strings=True)

    # Total loss
    loss = args.EETN_loss_coef * (2 * cycle_loss_A + cycle_loss_B) + 0.2 * (kl_loss_A + kl_loss_B)

    return loss


def train_iteration_ExtendedEETN(real_A, real_C, model_A, model_C, T_AB, T_BA, T_BC, T_CB, args):

    real_A, fp_mols_A = real_A
    real_C, fp_mols_C = real_C

    # embedder (METN)
    real_A_emb, kl_loss_A = model_A.forward_encoder(real_A)
    real_C_emb, kl_loss_C = model_C.forward_encoder(real_C)
    if args.kl_loss is False:
        kl_loss_A, kl_loss_C = None, None

    # prepare fingerprints
    if args.use_fp:
        if args.use_C_fp:
            real_A_fp_str = [smiles2fingerprint(model_A.tensor2string(mol), fp_translator=True) for mol in fp_mols_A]
            real_C_fp_str = [smiles2fingerprint(model_C.tensor2string(mol), fp_translator=True) for mol in fp_mols_C]
        else:
            real_A_fp_str = [smiles2fingerprint(model_A.tensor2string(mol), fp_translator=True) for mol in real_A]
            real_C_fp_str = [smiles2fingerprint(model_C.tensor2string(mol), fp_translator=True) for mol in real_C]
        real_A_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_A_fp_str]).to(device)
        real_C_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_C_fp_str]).to(device)
        real_A_fp = real_A_fp.detach()
        real_C_fp = real_C_fp.detach()
    else:
        real_A_fp, real_C_fp = None, None

    fake_B_emb = T_AB(real_A_emb, real_A_fp)
    fake_C_emb = T_BC(fake_B_emb, real_A_fp)
    cycle_B_emb = T_CB(fake_C_emb, real_A_fp)
    cycle_A_emb = T_BA(cycle_B_emb, real_A_fp)

    cycle_loss_A, _, recon_A = model_A.forward_decoder(real_A, cycle_A_emb, get_recon_strings=True)

    fake_B_emb = T_CB(real_C_emb, real_C_fp)
    fake_A_emb = T_BA(fake_B_emb, real_C_fp)
    cycle_B_emb = T_AB(fake_A_emb, real_C_fp)
    cycle_C_emb = T_BC(cycle_B_emb, real_C_fp)

    cycle_loss_C, _, recon_C = model_C.forward_decoder(real_C, cycle_C_emb, get_recon_strings=True)

    # Total loss
    loss = args.Extended_EETN_loss_coef * (2 * cycle_loss_A + cycle_loss_C) + 0.2 * (kl_loss_A + kl_loss_C)

    return loss


def validation(args, model_name, model_A, model_B, T_AB, model_C, T_BC, random_seed_list):
    # evaluation mode
    model_A.eval()
    model_B.eval()
    T_AB.eval()
    model_C.eval()
    T_BC.eval()

    # dataset loader
    valid_loader = get_dataloader(model_A, args, model_A.dataset.validset, batch_size=args.validation_batch_size, shuffle=False)

    # number samples in validset
    validset_len = len(model_A.dataset.validset)

    # tensor to molecule smiles
    def input_tensor2string(input_tensor):
        return model_A.tensor2string(input_tensor)

    trainset = set(model_A.dataset.get_trainset()).union(model_B.dataset.get_trainset()).union(model_C.dataset.get_trainset())

    #generate output molecule from input molecule
    def local_input2output(input_batch):
        return input2output(args, input_batch, model_A, T_AB, T_BC, model_C, random_seed_list, max_out_len=args.validation_max_len)

    # use general validation function
    avg_similarity, avg_property, avg_SR, avg_PL, avg_validity, avg_novelty, avg_diversity =\
        general_validation(args, local_input2output, input_tensor2string, valid_loader, validset_len, model_name, trainset)

    # back to train mode
    model_A.train()
    model_B.train()
    T_AB.train()
    model_C.train()
    T_BC.train()

    return avg_similarity, avg_property, avg_SR, avg_PL, avg_validity, avg_novelty, avg_diversity

def early_stop(early_stopping, current_criterion, best_criterion, runs_without_improvement):
    if early_stopping is not None:
        # first model or best model so far
        if best_criterion is None or current_criterion > best_criterion:
            runs_without_improvement = 0
        # no improvement
        else:
            runs_without_improvement += 1
        if runs_without_improvement >= early_stopping:
            return True, runs_without_improvement       # True = stop training
        else:
            return False, runs_without_improvement


if __name__ == "__main__":

    # parse arguments
    args = parse_arguments()

    # set seed
    set_seed(args.seed)

    # set device (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # disable rdkit error messages
    rdkit_no_error_print()

    # property_init
    property_init(args)

    if args.property is 'QED':
        embedder_epochs_num = 3
    elif args.property is 'DRD2':
        embedder_epochs_num = 27

    if args.no_pre_train_models or args.load_checkpoint:
        embedder_epochs_num = 0

    # prepare dataset
    dataset_file_A, dataset_file_B = create_dataset(args.property, args.rebuild_dataset, args)
    dataset_file_C = create_dynamic_dataset(args.property, args)

    dataset_C = Dataset(args, dataset_file_C, use_atom_tokenizer=args.tokenize, isA=False, isC=True, size=None)
    dataset_A = Dataset(args, dataset_file_A, use_atom_tokenizer=args.tokenize, isA=True, isC=False, size=dataset_C.get_size())
    dataset_B = Dataset(args, dataset_file_C, use_atom_tokenizer=args.tokenize, isA=False, isC=False, size=dataset_C.get_size())


    # create  and pre-train the embedders (METNs)
    model_A = Embedder(dataset_A, 'Embedder A', args).to(device)
    print("Fitting model_A")
    fit(args, model_A, embedder_epochs_num, is_validation=True)
    model_B = Embedder(dataset_B, 'Embedder B', args).to(device)
    print("Fitting model_B")
    fit(args, model_B, embedder_epochs_num, is_validation=False)
    model_C = Embedder(dataset_C, 'Embedder C', args).to(device)
    print("Fitting model_C")
    fit(args, model_C, embedder_epochs_num, is_validation=False)

    # create embedding translators (EETN)
    T_AB = Translator().to(device)
    T_BA = Translator().to(device)
    T_BC = Translator().to(device)
    T_CB = Translator().to(device)

    # weights
    T_AB.apply(weights_init_normal)
    T_BA.apply(weights_init_normal)
    T_BC.apply(weights_init_normal)
    T_CB.apply(weights_init_normal)

    # optimizer
    optimizer_T = torch.optim.Adam(itertools.chain(T_AB.parameters(), T_BA.parameters(), T_BC.parameters(), T_CB.parameters(),
                                                   get_model_train_params(model_A), get_model_train_params(model_B),
                                                   get_model_train_params(model_C)),
                                   lr=args.init_lr, betas=(0.5, 0.999))

    # scheduler
    lr_scheduler_T = torch.optim.lr_scheduler.LambdaLR(optimizer_T, lr_lambda=LambdaLR(args.epochs, args.epoch_init, args.epoch_decay).step)

    # train dataloaders
    A_train_loader = get_dataloader(model_A, args, model_A.dataset.get_trainset(), args.batch_size, collate_fn=None, shuffle=True)
    B_train_loader = get_dataloader(model_B, args, model_B.dataset.get_trainset(), args.batch_size, collate_fn=None, shuffle=True)
    C_train_loader = get_dataloader(model_C, args, model_C.dataset.get_trainset(), args.batch_size, collate_fn=None, shuffle=True)

    # generate random seeds
    random_seed_list = get_random_list(args.num_retries)


    ###### Training ######
    # update learning rate
    lr_scheduler_T.step()

    # for early stopping
    best_criterion = None
    runs_without_improvement = 0

    if args.load_checkpoint:
        T_AB, T_BA, model_A, model_B, T_BC, T_CB, model_C, best_criterion = load_checkpoint(args, device)

    for epoch in range(args.epoch_init, args.epochs + 1):
        print(' ')
        print('Training model - epoch #' + str(epoch))

        # statistics
        stats = Statistics()

        # set seed
        set_seed(args.seed + epoch)

        for _, (real_A, real_B, real_C) in enumerate(zip(A_train_loader, B_train_loader, C_train_loader)):

            optimizer_T.zero_grad()

            loss_AB = train_iteration_EETN(real_A, real_B, model_A, model_B, T_AB, T_BA, args)
            loss_BC = train_iteration_EETN(real_B, real_C, model_B, model_C, T_BC, T_CB, args)
            loss_AC = train_iteration_ExtendedEETN(real_A, real_C, model_A, model_C, T_AB, T_BA, T_BC, T_CB, args)
            loss = (loss_AB + loss_BC + loss_AC).sum()

            # update statistics
            stats.update_total_loss(loss, loss_AB, loss_AC, None, loss_BC)

            loss.backward()
            optimizer_T.step()

        # print epoch's statistics
        stats.print()

        # run validation
        if args.is_valid is True and (epoch == args.epoch_init or epoch % args.validation_freq == 0):
            avg_similarity, avg_property, avg_SR, avg_PL, avg_validity, avg_novelty, avg_diversity = \
                validation(args, 'Our AC', model_A, model_B, T_AB, model_C, T_BC, random_seed_list)


        # update learning rate
        lr_scheduler_T.step()

    save_checkpoint(model_A, model_B, model_C, T_AB, T_BA, T_BC, T_CB, args)

    print("Done Training")
