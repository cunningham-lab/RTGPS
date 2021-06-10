import numpy as np
import torch
import gpytorch
import time
from datetime import datetime
from experiments.load_data import load_uci_data
from experiments.experiment_fns import print_initial_hypers, get_hypers, get_string_time_taken, save_results
from experiments.experiment_fns import GPRegressionModel as GptGPRegressionModel

from rrcg.gp_utils import GPRegressionModel, MLL, GaussianLikelihood
from rrcg_experiments.experiment_fns_for_rrcg import TrainGPRegressionModel, print_dist_stats, \
    start_all_logging_instruments
from rrcg import rr_settings
import os
import fire


##################
# Setting Params #
##################
"""
train_n = 1000

method = 'rrcg'
num_cg = 10
num_cg_lanczos = num_cg
min_rr_iter = 20
max_rr_iter = train_n
temp = 0.05
rr_nsamples = 2

use_scheduler = True
total_iters = 1000
#total_iters = 500
#lr = 5e-2
lr = 3e-2a
#lr=1e-2
"""


def run(dataset, model_name, keops=True, total_n=-1):
    log_path = f'./logs/{dataset}'
    with open(os.path.join(log_path, model_name.replace("model", "loss") + ".log"), "r") as f:
        lines = f.read().split("\n")

    method = None
    kernel_type = None
    seed = None
    num_cg = None
    total_niters = None
    lr = None
    temp = None
    rr_iter_min = None
    dist_of_iter_mean = None
    dist_of_iter_var = None
    for line in lines:
        if "Hyper: method" in line:
            method = line.split("Hyper: method: ")[1]
        elif "Hyper: kernel_type" in line:
            kernel_type = line.split("Hyper: kernel_type: ")[1]
        elif "Hyper: seed" in line:
            seed = int(line.split("Hyper: seed: ")[1])
        elif "Hyper: num_cg" in line:
            num_cg = line.split("Hyper: num_cg: ")[1]
        elif "Hyper: total_niters" in line:
            total_niters = line.split("Hyper: total_niters: ")[1]
        elif "Hyper: lr" in line:
            lr = line.split("Hyper: lr: ")[1]
        elif "Hyper: temp" in line:
            temp = line.split("Hyper: temp: ")[1]
        elif "Hyper: rr_iter_min" in line:
            rr_iter_min = line.split("Hyper: rr_iter_min: ")[1]
        elif "Hyper: dist_of_iter_mean" in line:
            dist_of_iter_mean = line.split("Hyper: dist_of_iter_mean: ")[1]
        elif "Hyper: dist_of_iter_var" in line:
            dist_of_iter_var = line.split("Hyper: dist_of_iter_var: ")[1]

    assert method in ['cholesky', 'rrcg', 'cg', 'gpt-cholesky', 'gpt-cg'], method
    assert kernel_type in ['rbf', 'matern05', 'matern15', 'mater25', 'rbf-ard'], kernel_type
    assert seed is not None

    torch.manual_seed(seed)
    np.random.seed(seed)

    ###############
    ##### Data ####
    ###############
    train_x, train_y, test_x, test_y, valid_x, valid_y = load_uci_data(data_dir=None, dataset=dataset, total_n=total_n,
                                                                       cuda=torch.cuda.is_available(), verbose=True)
    print("Dataset stats:")
    train_n, valid_n, test_n = len(train_x), len(valid_x), len(test_x)
    print("train_n = {}, val_n = {}, test_n = {}".format(train_n, valid_n, test_n))

    ###############
    #### Model ####
    ###############
    if method == 'gpt-cholesky' or method == 'gpt-cg':
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GptGPRegressionModel(
            train_x, train_y, likelihood, kernel_type=kernel_type, use_keops=keops) #torch.cuda.is_available())
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    else:
        likelihood = GaussianLikelihood()
        model = GPRegressionModel(
            train_x, train_y, likelihood, kernel_type=kernel_type, use_keops=keops) #torch.cuda.is_available())
        mll = MLL(likelihood, model)

    if torch.cuda.is_available():
        model.cuda()
    get_ls = False if kernel_type == 'rbf-ard' else True
    print_initial_hypers(model, print_ls=get_ls)

    ####################
    ##### Training #####
    ####################
    train_start = time.time()

    experiment_params = {
        # dataset params:
        'dataset': dataset,
        'train_n': train_n,
        'valid_n': valid_n,
        'test_n': test_n,

        # model params
        'method': method,
        'kernel_type': kernel_type,
        'keops': keops,

        # training params
        'seed': seed,

        # other args
        'num_cg': num_cg,
        'total_niters': total_niters,
        'lr': lr,
        'temp': temp,
        'rr_iter_min': rr_iter_min,
        'dist_of_iter_mean': dist_of_iter_mean,
        'dist_of_iter_var': dist_of_iter_var,
    }

    log_path = f'./logs/{dataset}'
    if not os.path.exists(log_path):
        print(f"Creating log_path: {log_path}")
        os.makedirs(log_path)
    time_stamp = model_name.split("model_")[1]
    logger = start_all_logging_instruments(experiment_params, results_path=log_path+f'/{method}_eval_', time_stamp=time_stamp)
    logger.info("Evaluating")

    model_path = f'./models/{dataset}'
    state_dict = torch.load(os.path.join(model_path, model_name))
    model.load_state_dict(state_dict)

    ##############
    # Prediction #
    ##############
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    eval_valid_start = time.time()
    valid_rmse, valid_nll = get_prediction_stats(model, valid_x, valid_y)
    eval_valid_end = time.time()
    logger.info(
        "Valid rmse = {:.4f}, nll = {:.4f}".format(valid_rmse.cpu().numpy(), torch.mean(valid_nll).cpu().numpy()))
    logger.info(get_string_time_taken(eval_valid_start, eval_valid_end))

    eval_test_start = time.time()
    test_rmse, test_nll = get_prediction_stats(model, test_x, test_y)
    logger.info(
        "Test rmse = {:.4f}, nll = {:.4f}".format(test_rmse.cpu().numpy(), torch.mean(test_nll).cpu().numpy()))
    eval_test_end = time.time()
    logger.info(get_string_time_taken(eval_test_start, eval_test_end))


def get_prediction_stats(model, x, y):
    with torch.no_grad():
        with gpytorch.settings.fast_pred_var():
            observed_pred = model.likelihood(model(x))
            rmse = torch.sqrt(torch.mean(torch.pow(observed_pred.mean - y, 2)))  # a scalar

            if isinstance(model, GPRegressionModel):
                nll = -model.likelihood.log_marginal_from_marginal(marginal=observed_pred, observations=y)  # (bsz_x, )
            else:
                nll = -model.likelihood.log_marginal(observations=y, function_dist=model(x))

    return rmse, nll


if __name__ == "__main__":
    fire.Fire(run)
