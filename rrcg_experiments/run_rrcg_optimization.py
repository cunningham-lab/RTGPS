import numpy as np
import torch
import gpytorch
import time
from datetime import datetime

from experiments.load_data import load_uci_data
from rrcg.gp_utils import GPRegressionModel, MLL, GaussianLikelihood
from rrcg import rr_settings
from rrcg_experiments.experiment_fns_for_rrcg import TrainGPRegressionModel, print_dist_stats, \
    start_all_logging_instruments, get_dist_of_iter, \
    print_initial_hypers, get_hypers, get_string_time_taken, save_results
from rrcg_experiments.experiment_fns_for_rrcg import GPRegressionModel as GptGPRegressionModel

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


def run(total_n, method, dataset, kernel_type='rbf', num_cg=10,
        rr_nsamples=2, use_scheduler=True, seed=10, total_iters=1000, lr=3e-2, eval=True, keops=True,
        save_model=False, save_hyper_trace=False, load=None, fix_outputscale=False, output_scale=0.62923026, **dist_of_iter_kwargs):
    assert method in ['cholesky', 'rrcg', 'cg', 'gpt-cholesky', 'gpt-cg'], method
    assert kernel_type in ['rbf', 'matern05', 'matern15', 'mater25', 'rbf-ard'], kernel_type

    torch.manual_seed(seed)
    np.random.seed(seed)

    num_cg_lanczos = num_cg

    ###############
    ##### Data ####
    ###############
    train_x, train_y, test_x, test_y, valid_x, valid_y = load_uci_data(data_dir=None, dataset=dataset, total_n=total_n,
                                                                       cuda=torch.cuda.is_available(), verbose=True)
    print("Dataset stats:")
    train_n, valid_n, test_n = len(train_x), len(valid_x), len(test_x)
    print("train_n = {}, val_n = {}, test_n = {}".format(train_n, valid_n, test_n))
    if dist_of_iter_kwargs.get("rr_iter_max") is None:
        dist_of_iter_kwargs['rr_iter_max'] = train_n

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

    if fix_outputscale:
        hypers = {
            'covar_module.outputscale': output_scale,
        }
        model.initialize(**hypers)

    if torch.cuda.is_available():
        model.cuda()
    get_ls = False if kernel_type == 'rbf-ard' else True
    print_initial_hypers(model, print_ls=get_ls)

    if method == 'rrcg':
        dist_of_iter = get_dist_of_iter(**dist_of_iter_kwargs)
        print_dist_stats(dist_of_iter)
    else:
        dist_of_iter = None

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
        'num_cg': num_cg,
        'rr_nsamples': rr_nsamples,
        'dist_of_iter_mean': dist_of_iter.mean if dist_of_iter else None,
        'dist_of_iter_var': dist_of_iter.var if dist_of_iter else None,
        'fix_outputscale': fix_outputscale,
        'output_scale': output_scale,
        **dist_of_iter_kwargs,

        # training params
        'seed': seed,
        'total_niters': total_iters,
        'lr': lr,
        'use_scheduler': use_scheduler,

        # misc
        'save_model': save_model
    }

    log_path = f'./logs/{dataset}'
    if not os.path.exists(log_path):
        print(f"Creating log_path: {log_path}")
        os.makedirs(log_path)
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logger = start_all_logging_instruments(experiment_params, results_path=log_path+f'/{method}_', time_stamp=time_stamp)

    if load:
        state_dict = torch.load(load)
        current_state_dict = model.state_dict()
        logger.info(f"Loading parameters from {load}")
        for name, param in state_dict.items():
            if name in current_state_dict:
                logger.info(f"{name}: {param.shape} into {current_state_dict[name].shape}")
        model.load_state_dict(state_dict, strict=False)

    # hyper-param-trace hyper_trace path
    if save_hyper_trace:
        if not os.path.exists("./hyper_trace/{}".format(dataset)):
            print("Creating hyper_trace/{} directory...".format(dataset))
            os.makedirs("./hyper_trace/{}".format(dataset))

        if method == 'cholesky' or method == 'gpt-cholesky':
            hyper_trace_results_path = \
                f'./hyper_trace/{dataset}/method={method}_kernel={kernel_type}_ndata={total_n}_lr={lr}_niters={total_iters}_schedule={use_scheduler}_seed={seed}.pkl'
        elif method == 'cg' or method == 'gpt-cg':
            hyper_trace_results_path = \
                f'./hyper_trace/{dataset}/method={method}_numcg={num_cg}_kernel={kernel_type}_ndata={total_n}_lr={lr}_niters={total_iters}_schedule={use_scheduler}_seed={seed}.pkl'
        else:
            hyper_trace_results_path = \
                f'./hyper_trace/{dataset}/method={method}_mean={torch.round(dist_of_iter.mean)}_std={torch.round(dist_of_iter.std)}_rrnsamples={rr_nsamples}_kernel={kernel_type}_ndata={total_n}_lr={lr}_niters={total_iters}_schedule={use_scheduler}_seed={seed}.pkl'
        print("Saving hyper_trace to \n", hyper_trace_results_path)

    if save_model:
        model_path = f'./models/{dataset}'
        if not os.path.exists(model_path):
            print(f"Creating model_path: {model_path}")
            os.makedirs(model_path)
        save_path = os.path.join(model_path, f'{method}_model_{time_stamp}')
    else:
        save_path = None

    tr = TrainGPRegressionModel(model, likelihood, mll=mll, dist_of_iter=dist_of_iter, logger=logger, method=method,
                                total_iters=total_iters, lr=lr, use_scheduler=use_scheduler, track_ls=get_ls,
                                track_hyper_trace=save_hyper_trace,
                                fix_outputscale=fix_outputscale, save_path=save_path)

    if method == 'cholesky':
        with gpytorch.settings.max_cholesky_size(1e10):
            tr.train()
    else:
        with gpytorch.settings.max_cholesky_size(0):
            with gpytorch.settings.cg_tolerance(1e-8):
                if method == 'rrcg':
                    with rr_settings.use_rr_cg():
                        with rr_settings.rr_cg_nsamples(rr_nsamples):
                            tr.train()
                else:
                    with gpytorch.settings.max_cg_iterations(num_cg):
                        with gpytorch.settings.max_lanczos_quadrature_iterations(num_cg_lanczos):
                            tr.train()
    train_end = time.time()

    logger.info(f"\nFinish Training.")
    logger.info(get_string_time_taken(train_start, train_end))
    logger.info(get_hypers(model))

    if save_hyper_trace:
        save_results(tr.hyper_trace, hyper_trace_results_path)

    if save_model:
        torch.save(model.state_dict(), save_path)


    ##############
    # Prediction #
    ##############
    if eval:
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
