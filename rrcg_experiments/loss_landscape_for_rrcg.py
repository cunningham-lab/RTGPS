import fire
import os
import math
import torch
import gpytorch
import tqdm
import pandas as pd
import numpy as np
from itertools import product

from experiments.experiment_fns import set_hypers
from experiments.load_data import load_uci_data

from rrcg.gp_utils import GaussianLikelihood, GPRegressionModel, MLL
from rrcg import rr_settings
from rrcg.dist_of_iterations_for_rrcg import ExpDecayDist



def mll_sweep(
    data_dir, save_dir, method='rrcg',
    lengthscale_min=0.01, lengthscale_max=1., noise_min=0.01, noise_max=1., grid_size=25,
    outputscale=1., overwrite_result=False, seed=10, num_cg=None, rr_temp=0.05, rr_min=20, rr_max=500,
        rr_nsamples=2, nx=-1,
):
    assert method in ['rrcg', 'cg', 'cholesky'], \
        "method muse be chosen among rrcg, cg, cholesky, but got {}".format(method)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("data_dir = ", data_dir)
    print("save_dir = ", save_dir)
    train_x, train_y, *_ = load_uci_data(data_dir, "pol")


    if nx == -1:
        nx = train_x.shape[0]
    else:
        assert nx > 0 and nx <= train_x.shape[0], nx
        train_x = train_x[:nx]
        train_y = train_y[:nx]

    if rr_max > nx:
        rr_max = nx
    if rr_min > nx:
        rr_min = nx

    likelihood = GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)

    #mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    mll = MLL(likelihood, model)

    # Maybe cuda
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()  # Contains likelihood

    dist_of_iter = None
    if method == 'rrcg':
        # currently, fixing dist_of_iter
        dist_of_iter = ExpDecayDist(temp=rr_temp, min=rr_min, max=rr_max, dtype=torch.float,
                                    device=train_x.device)
        print("\ndist of iter mean = {}".format(dist_of_iter.mean))
        print("dist of iter std = {}\n".format(np.sqrt(dist_of_iter.var)))

    # Maybe get existing hyper_trace?
    results_path = os.path.join(save_dir, "{}_loss_landscape.csv".format(method))
    #results_path = os.path.join(save_dir, "rrcg_5samples.csv")
    if os.path.exists(results_path):
        if not overwrite_result:
            print("Loading previous hyper_trace...")
            results = pd.read_csv(results_path).values.tolist()
        else:
            print("Overwrite previous hyper_trace...")
            results = []
    else:
        results = []

    # Values to cycle through
    lengthscales = torch.linspace(lengthscale_min, lengthscale_max, grid_size).tolist()
    noises = torch.linspace(noise_min, noise_max, grid_size).tolist()

    print("ls max ", np.max(lengthscales))
    print("noises max ", np.max(noises))

    iterator = tqdm.tqdm(product(lengthscales, noises), desc="Hyperparam configs", total=(grid_size ** 2))


    # Loop
    for idx, (lengthscale, noise) in enumerate(iterator):
        model.train()
        set_hypers(model, noise, lengthscale, outputscale)
        if method == 'cholesky':
            with gpytorch.settings.max_cholesky_size(10000000):
                mll_value = mll(model(train_x), train_y, dist_of_iter=dist_of_iter).item()
        else:
            with gpytorch.settings.max_cholesky_size(0):
                if method == 'cg':
                    with gpytorch.settings.max_cg_iterations(num_cg):
                        with gpytorch.settings.max_lanczos_quadrature_iterations(num_cg):
                            mll_value = mll(model(train_x), train_y, dist_of_iter=dist_of_iter).item()
                else:
                     # rrcg
                    with rr_settings.use_rr_cg():
                        with rr_settings.rr_cg_nsamples(rr_nsamples):
                            mll_value = mll(model(train_x), train_y, dist_of_iter=dist_of_iter).item()

        iterator.set_postfix(
            mll=mll_value,
            os=model.covar_module.outputscale.item(),
            ls=model.covar_module.base_kernel.lengthscale.item(),
            noise=model.likelihood.noise.item(),
        )

        # Log hyper_trace
        results.append((lengthscale, noise, outputscale, mll_value))
        if not (idx + 1) % 25 or (idx == len(iterator) - 1):
            df = pd.DataFrame(results, columns=["lengthscale", "noise", "outputscale", "mll"])
            df.to_csv(results_path, index=False)

if __name__ == "__main__":
    fire.Fire(mll_sweep)
