import fire
import os
import math
import torch
import gpytorch
import tqdm
import pandas as pd
import numpy as np
from itertools import product

from experiments.experiment_fns import GPRegressionModel, train, set_hypers
from experiments.load_data import load_uci_data


def mll_sweep(
    data_dir, save_dir,
    lengthscale_min=0.01, lengthscale_max=1., noise_min=0.01, noise_max=0.2, grid_size=25,
    outputscale=1.,
):
    train_x, train_y, *_ = load_uci_data(data_dir, "pol")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Maybe cuda
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()  # Contains likelihood

    # Values to cycle through
    lengthscales = torch.linspace(lengthscale_min, lengthscale_max, grid_size).tolist()
    noises = torch.linspace(noise_min, noise_max, grid_size).tolist()
    iterator = tqdm.tqdm(product(lengthscales, noises), desc="Hyperparam configs", total=(grid_size ** 2))

    # Maybe get existing results?
    results_path = os.path.join(save_dir, "exact_loss_landscape.csv")
    if os.path.exists(results_path):
        results = pd.read_csv(results_path).values.tolist()
    else:
        results = []

    # Loop
    for idx, (lengthscale, noise) in enumerate(iterator):
        model.train()
        set_hypers(model, noise, lengthscale, outputscale)
        mll_value = mll(model(train_x), train_y).item()
        iterator.set_postfix(
            mll=mll_value,
            os=model.covar_module.outputscale.item(),
            ls=model.covar_module.base_kernel.lengthscale.item(),
            noise=model.likelihood.noise.item(),
        )

        # Log results
        results.append((lengthscale, noise, outputscale, mll_value))
        if not (idx + 1) % 25 or (idx == len(iterator) - 1):
            df = pd.DataFrame(results, columns=["lengthscale", "noise", "outputscale", "mll"])
            df.to_csv(results_path, index=False)


if __name__ == "__main__":
    fire.Fire(mll_sweep)
