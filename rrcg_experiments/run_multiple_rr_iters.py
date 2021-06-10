#TODO: run multiple RR optimizatin with different number of rr-cg iterations
# compute the exact loss likelihood, as a function of number of optimization iterations
# also test different learning rate
# also test the variance of rr distribution

# and compare against CG in terms of rr iters and number of optimization iterations
# and compare to Cholesky

# collect hyperparameter trajectory
# use another function to compute the loss (using cholesky evaluation)

import torch
import gpytorch
import numpy as np
import pickle
import os
import fire

from experiments.load_data import load_uci_data
from rrcg.gp_utils import GaussianLikelihood, GPRegressionModel, MLL
from rrcg_experiments.experiment_fns_for_rrcg import set_hypers


def get_hyper_traces_and_params(dataset, file_name, base_dir="./hyper_trace"):
    with open(os.path.join(base_dir, dataset, file_name), "rb") as f:
        hyper_traces = pickle.load(f)

    stripped_file_name = file_name[:-4] # remove pickle
    print("stripeed file name = ", stripped_file_name)
    exp_hyper_params = dict(item.split("=") for item in stripped_file_name.split("_"))
    exp_hyper_params['dataset'] = dataset
    print("exp_hyper_params = ")
    print(exp_hyper_params)
    return hyper_traces, exp_hyper_params


def get_exact_loss_from_hyper_traces(hyper_traces, exp_hyper_params, save_path, keops=True, return_opt_loss=False, return_exact_loss=False,
                                     eval_method='cholesky'):
    # use cholesky method to plot

    seed = int(exp_hyper_params['seed'])
    kernel_type = exp_hyper_params['kernel']
    dataset = exp_hyper_params['dataset']
    total_n = int(exp_hyper_params['ndata'])

    torch.manual_seed(seed)
    np.random.seed(seed)

    use_cuda = torch.cuda.is_available()
    # load data by seed
    train_x, train_y, test_x, test_y, valid_x, valid_y = load_uci_data(data_dir=None, dataset=dataset, total_n=total_n,
                                                                       cuda=use_cuda, verbose=True)

    ls_trace, os_trace, noise_trace = hyper_traces['ls'], hyper_traces['os'], hyper_traces['noise']

    likelihood = GaussianLikelihood()
    model = GPRegressionModel(
        train_x, train_y, likelihood, kernel_type=kernel_type, use_keops=keops)  # torch.cuda.is_available())
    mll = MLL(likelihood, model)
    if use_cuda:
        model.cuda()

    n_iters = len(ls_trace)
    exact_loss_trace = np.zeros(n_iters)
    model.train()

    with torch.no_grad():
        # decide using cholesky or CG
        if eval_method == 'cholesky':
            with gpytorch.settings.max_cholesky_size(1e10):
                for i, (ls, output_scale, noise) in enumerate(zip(ls_trace, os_trace, noise_trace)):

                    set_hypers(model, noise_scale=noise, ls=ls, output_scale=output_scale)
                    y_hat = model(train_x)
                    loss = -mll(y_hat, train_y)
                    exact_loss_trace[i] = loss.item()
        else:
            with gpytorch.settings.max_cg_iterations(200):
                with gpytorch.settings.max_lanczos_quadrature_iterations(200):
                    with gpytorch.settings.cg_tolerance(1e-6):
                        for i, (ls, output_scale, noise) in enumerate(zip(ls_trace, os_trace, noise_trace)):
                            set_hypers(model, noise_scale=noise, ls=ls, output_scale=output_scale)
                            y_hat = model(train_x)
                            loss = -mll(y_hat, train_y)
                            exact_loss_trace[i] = loss.item()

    opt_loss = hyper_traces.get("loss", None)

    with open(save_path, "wb") as f:
        dict_to_save = dict(exact_loss=exact_loss_trace, opt_loss=opt_loss)
        pickle.dump(dict_to_save, f)

    exact_loss = None
    if not return_opt_loss:
        opt_loss = None
    if return_exact_loss:
        exact_loss = exact_loss_trace
    return opt_loss, exact_loss


def main(hyper_trace_dir, keops=True):
    all_datasets = ['3droad', 'bike', 'buzz', 'elevators', 'keggdirected', 'keggundirected', 'kin40k', 'pol', 'protein',
                    'slice', 'song']
    cholesky_available_datasets = ['pol', 'elevators', 'bike']

    print("\nStart running run_multiple_rr_iters.py ...\n")
    for dataset in os.listdir(hyper_trace_dir):
        if dataset in all_datasets:
            print(f"\n############## Dataset {dataset} #############")
            for file_name in os.listdir(os.path.join(hyper_trace_dir, dataset)):
                if '.pkl' in file_name:
                    print(f"file_name = {file_name}")

                    if dataset in cholesky_available_datasets:
                        eval_method = 'cholesky'
                    else:
                        eval_method = 'cg'

                    hyper_traces, exp_hyper_params = get_hyper_traces_and_params(dataset=dataset, file_name=file_name, base_dir=hyper_trace_dir)
                    save_dir = f"./exact_loss/{dataset}"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, file_name)
                    _, _ = get_exact_loss_from_hyper_traces(hyper_traces, exp_hyper_params, save_path, keops=keops,
                                                            eval_method=eval_method)


if __name__ == "__main__":
    fire.Fire(main)

    """
    dataset = 'pol'
    file_name = "method=cg_numcg=20_kernel=rbf_ndata=100_lr=0.05_niters=10_schedule=True_seed=10.pkl"

    hyper_traces, exp_hyper_params = get_hyper_traces_and_params(dataset=dataset, file_name=file_name)

    save_dir = f"./exact_loss/{dataset}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    opt_loss, exact_loss = get_exact_loss_from_hyper_traces(hyper_traces, exp_hyper_params, save_path, keops=True, return_opt_loss=True, return_exact_loss=True)
    """
