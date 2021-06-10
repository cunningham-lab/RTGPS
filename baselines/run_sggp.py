import numpy as np
import torch
import gpytorch
import time
from datetime import datetime
from experiments.load_data import load_uci_data
from experiments.experiment_fns import print_initial_hypers, get_hypers, get_string_time_taken, save_results
from rrcg_experiments.experiment_fns_for_rrcg import start_all_logging_instruments
from scipy.cluster.vq import kmeans2
import faiss
import tqdm
import math

import os
import fire


##################
# Setting Params #
##################
"""
total_iters = 300
k = k
"""
class SGGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, k=16, kernel_type='rbf'):
        super().__init__(train_x, train_y, likelihood)
        self.k = k
        self.res = faiss.StandardGpuResources()
        self.register_buffer("train_x", train_x)

        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_type == 'rbf':
            kernel = gpytorch.kernels.keops.RBFKernel()
        elif kernel_type == 'matern05':
            kernel = gpytorch.kernels.keops.MaternKernel(nu=0.5)
        elif kernel_type == 'matern15':
            kernel = gpytorch.kernels.keops.MaternKernel(nu=1.5)
        elif kernel_type == 'matern25':
            kernel = gpytorch.kernels.keops.MaternKernel(nu=2.5)
        elif kernel_type == 'rbf-ard':
            ard_dim = train_x.shape[-1]
            print(f"Getting ard_dim from training input, ard_dim = {ard_dim}")
            kernel = gpytorch.kernels.keops.RBFKernel(ard_num_dims=ard_dim)
        else:
            raise ValueError("kernel_type must be chosen among {}, but got {}.".format(
                ['rbf', 'matern05', 'matern15', 'matern25'], kernel_type))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

        def scale(grad):
            return grad * (self.k / math.log(k))
        self.covar_module.raw_outputscale.register_hook(scale)

    def compute_train_nn_idx(self):
        assert self.k > 0
        x = (self.train_x.data.float() / self.covar_module.base_kernel.lengthscale.data.float()).cpu().numpy()
        self.cpu_index = faiss.IndexFlatL2(self.train_x.size(-1))
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        self.gpu_index.add(x)
        self.train_nn_idx = torch.from_numpy(self.gpu_index.search(x, self.k)[1]).long().cuda()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def run(dataset, total_n=-1, kernel_type='rbf', num_inducing=1024, k=16,
        use_scheduler=True, seed=10, total_iters=20, lr=0.01, eval=True,
        save_model=True, batch_size=1024):
    assert kernel_type in ['rbf', 'matern05', 'matern15', 'matern25', 'rbf-ard'], kernel_type

    torch.manual_seed(seed)
    np.random.seed(seed)

    ###############
    ##### Data ####
    ###############
    train_x, train_y, test_x, test_y, valid_x, valid_y = load_uci_data(data_dir=None, dataset=dataset, total_n=total_n,
                                                                       cuda=torch.cuda.is_available(), verbose=True)

    # Create batches of data


    print("Dataset stats:")
    train_n, valid_n, test_n = len(train_x), len(valid_x), len(test_x)
    print("train_n = {}, val_n = {}, test_n = {}".format(train_n, valid_n, test_n))

    ###############
    #### Model ####
    ###############
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SGGPModel(train_x, train_y, likelihood, kernel_type=kernel_type, k=k)

    if torch.cuda.is_available():
        model.cuda()
    get_ls = False if kernel_type == 'rbf-ard' else True
    print_initial_hypers(model, print_ls_flag=get_ls)

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
        'method': "sggp",
        'kernel_type': kernel_type,
        'k': k,

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
    logger = start_all_logging_instruments(experiment_params, results_path=log_path+f'/sggp_', time_stamp=time_stamp)

    # Training loop
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(total_iters * 0.5), int(total_iters * 0.75)], gamma=0.1
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for iteration in range(total_iters):
        model.compute_train_nn_idx()
        batch_order = torch.randperm(len(train_x), device=train_x.device)

        iterator = tqdm.tqdm(batch_order.split(batch_size), disable=not(os.getenv("TQDM")))
        for index in iterator:
            indices = model.train_nn_idx[index]
            x_batch = train_x[indices]
            y_batch = train_y[indices]
            model.set_train_data(x_batch, y_batch, strict=False)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch).mean()
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=loss.item())

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

    model.set_train_data(train_x, train_y, strict=False)
    train_end = time.time()

    logger.info(f"\nFinish Training.")
    logger.info(get_string_time_taken(train_start, train_end))
    logger.info(get_hypers(model))

    if save_model:
        model_path = f'./models/{dataset}'
        if not os.path.exists(model_path):
            print(f"Creating model_path: {model_path}")
            os.makedirs(model_path)
        torch.save(model.state_dict(), os.path.join(model_path, f'sggp_model_{time_stamp}'))


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
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(x)
        posterior = gpytorch.distributions.MultivariateNormal(posterior.mean, posterior.lazy_covariance_matrix.evaluate_kernel())
        rmse = torch.sqrt(torch.mean(torch.pow(posterior.mean - y, 2)))  # a scalar
        nll = -model.likelihood.log_marginal(observations=y, function_dist=posterior)

    return rmse, nll


if __name__ == "__main__":
    fire.Fire(run)
