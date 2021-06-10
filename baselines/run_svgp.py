import numpy as np
import torch
import gpytorch
import time
from datetime import datetime
from experiments.load_data import load_uci_data
from experiments.experiment_fns import print_initial_hypers, get_hypers, get_string_time_taken, save_results
from rrcg_experiments.experiment_fns_for_rrcg import start_all_logging_instruments
from scipy.cluster.vq import kmeans2

import os
import fire


##################
# Setting Params #
##################
"""
total_iters = 300
batch_size = 1024
num_inducing = 1024
"""
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, likelihood, num_inducing=1024, kernel_type='rbf'):
        # Create initial inducing points
        train_x = train_x.detach().view(-1, train_x.size(-1))
        inducing_points = torch.randn(num_inducing, train_x.size(-1), dtype=train_x.dtype)
        inducing_points = torch.tensor(
            kmeans2(train_x.cpu().numpy(), inducing_points.numpy(), minit='matrix')[0]
        ).to(train_x.device)

        # Create variational objects
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )

        super().__init__(variational_strategy)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_type == 'rbf':
            kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == 'matern05':
            kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        elif kernel_type == 'matern15':
            kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_type == 'matern25':
            kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel_type == 'rbf-ard':
            ard_dim = train_x.shape[-1]
            print(f"Getting ard_dim from training input, ard_dim = {ard_dim}")
            kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dim)
        else:
            raise ValueError("kernel_type must be chosen among {}, but got {}.".format(
                ['rbf', 'matern05', 'matern15', 'matern25'], kernel_type))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def run(dataset, total_n=-1, kernel_type='rbf', num_inducing=1024, batch_size=1024,
        use_scheduler=True, seed=10, total_iters=300, lr=0.01, eval=True, load=None,
        save_model=True):
    assert kernel_type in ['rbf', 'matern05', 'matern15', 'mater25', 'rbf-ard'], kernel_type

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
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SVGPModel(train_x, train_y, likelihood, kernel_type=kernel_type, num_inducing=num_inducing)

    if torch.cuda.is_available():
        model.cuda()
    get_ls = False if kernel_type == 'rbf-ard' else True
    print_initial_hypers(model, print_ls_flag=get_ls)

    if load is not None:
        print("Loading")
        state_dict = torch.load(load)
        model.load_state_dict(state_dict)

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
        'method': "svgp",
        'kernel_type': kernel_type,
        'num_inducing': num_inducing,

        # training params
        'seed': seed,
        'total_niters': total_iters,
        'batch_size': batch_size,
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
    logger = start_all_logging_instruments(experiment_params, results_path=log_path+f'/svgp_', time_stamp=time_stamp)

    # Create dataloader
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(total_iters * 0.5), int(total_iters * 0.75)], gamma=0.1
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_n)

    for iteration in range(total_iters):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()
        outputscale = model.covar_module.outputscale.item()
        noise = model.likelihood.noise.item()
        time_taken = time.time() - train_start
        if model.covar_module.base_kernel.lengthscale.numel() == 1:
            ls = model.covar_module.base_kernel.lengthscale.item()
            text = f'iter: {iteration:4d} | loss: {loss.item():+1.4e} | lr: {lr:.5f} | '
            text += f'ls: {ls:4.6f} | noise: {noise:4.6f} | os: {outputscale:4.6f} | '
            text += f'time: {time_taken:4.2f} sec'
        else:
            text = f'iter: {iteration:4d} | loss: {loss.item():+1.4e} | lr: {lr:.5f} | '
            text += f'noise: {noise:4.6f} | os: {outputscale:4.6f} | '
            text += f'time: {time_taken:4.2f} sec'
        logger.info(text)

    train_end = time.time()

    logger.info(f"\nFinish Training.")
    logger.info(get_string_time_taken(train_start, train_end))
    logger.info(get_hypers(model))

    if save_model:
        model_path = f'./models/{dataset}'
        if not os.path.exists(model_path):
            print(f"Creating model_path: {model_path}")
            os.makedirs(model_path)
        torch.save(model.state_dict(), os.path.join(model_path, f'svgp_model_{time_stamp}'))

    ##############
    # Prediction #
    ##############
    if eval:
        model.eval()
        likelihood.eval()

        # Make predictions by feeding model through likelihood
        eval_valid_start = time.time()
        valid_mse = 0.
        valid_nll = 0.
        for x_batch, y_batch in zip(valid_x.split(512), valid_y.split(512)):
            batch_rmse, batch_nll = get_prediction_stats(model, x_batch, y_batch)
            valid_mse += batch_rmse.pow(2).mul(len(y_batch))
            valid_nll += batch_nll.sum()
        valid_rmse = (valid_mse / len(valid_y)).sqrt()
        valid_nll = valid_nll / len(valid_y)
        eval_valid_end = time.time()
        logger.info(
            "Valid rmse = {:.4f}, nll = {:.4f}".format(valid_rmse.cpu().numpy(), torch.mean(valid_nll).cpu().numpy()))
        logger.info(get_string_time_taken(eval_valid_start, eval_valid_end))

        eval_test_start = time.time()
        test_mse = 0.
        test_nll = 0.
        for x_batch, y_batch in zip(test_x.split(512), test_y.split(512)):
            batch_rmse, batch_nll = get_prediction_stats(model, x_batch, y_batch)
            test_mse += batch_rmse.pow(2).mul(len(y_batch))
            test_nll += batch_nll.sum()
        test_rmse = (test_mse / len(test_y)).sqrt()
        test_nll = test_nll / len(test_y)
        logger.info(
            "Test rmse = {:.4f}, nll = {:.4f}".format(test_rmse.cpu().numpy(), torch.mean(test_nll).cpu().numpy()))
        eval_test_end = time.time()
        logger.info(get_string_time_taken(eval_test_start, eval_test_end))


def get_prediction_stats(model, x, y):
    with torch.no_grad():
        observed_pred = model.likelihood(model(x))
        rmse = torch.sqrt(torch.mean(torch.pow(observed_pred.mean - y, 2)))  # a scalar
        nll = -model.likelihood.log_marginal(observations=y, function_dist=model(x))

    return rmse, nll


if __name__ == "__main__":
    fire.Fire(run)
