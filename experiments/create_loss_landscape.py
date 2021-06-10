import os
import torch
import gpytorch
import tqdm
import pandas as pd
from itertools import product
from experiments.experiment_fns import GPRegressionModel, set_hypers
from experiments.load_data import load_uci_data_ap


def mll_sweep(data_dir, save_dir,
              lengthscale_min=0.01, lengthscale_max=1., noise_min=0.01, noise_max=0.2,
              grid_size=25, outputscale=1.):
    train_x, train_y, *_ = load_uci_data_ap(data_dir, 'pol')
    train_n = int(1.e2)
    train_x, train_y = train_x[:train_n, :], train_y[:train_n]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()

    lengthscales = torch.linspace(lengthscale_min, lengthscale_max, grid_size).tolist()
    noises = torch.linspace(noise_min, noise_max, grid_size).tolist()
    iterator = tqdm.tqdm(product(lengthscales, noises),
                         desc="Hyperparam configs", total=(grid_size ** 2))

    results_path = os.path.join(save_dir,
                                'exact_loss_landscape_' + str(train_n) + '.csv')
    results = []

    for idx, (lengthscale, noise) in enumerate(iterator):
        model.train()
        set_hypers(model, noise, lengthscale, outputscale)
        mll_value = mll(model(train_x), train_y).item()
        iterator.set_postfix(mll=mll_value,
                             os=model.covar_module.outputscale.item(),
                             ls=model.covar_module.base_kernel.lengthscale.item(),
                             noise=model.likelihood.noise.item())

        results.append((lengthscale, noise, outputscale, mll_value))
    df = pd.DataFrame(results, columns=["lengthscale", "noise", "outputscale", "mll"])
    df.to_csv(results_path, index=False)


if __name__ == "__main__":
    lengthscale_min = 0.01
    lengthscale_max = 1.
    noise_min = 0.01
    noise_max = 1.
    grid_size = 20
    outputscale = 0.62923026
    data_dir = './datasets'
    save_dir = './results/'
    mll_sweep(data_dir, save_dir,
              lengthscale_min, lengthscale_max,
              noise_min, noise_max,
              grid_size, outputscale)
