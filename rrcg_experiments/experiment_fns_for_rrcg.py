from tqdm import tqdm
import pickle
import time
from datetime import datetime
import logging
import numpy as np
import torch
import gpytorch as gpt

from rrcg.dist_of_iterations_for_rrcg import RRDist, ExpDecayDist, OneOverJ, Geometric


def update_iterator_print(iterator, loss, ls, os, noise, lr):
    iterator.set_postfix(loss=loss, ls=ls, os=os, noise=noise, lr=lr)


class TrainGPRegressionModel:

    def __init__(self, model, likelihood, mll, dist_of_iter, logger, method, total_iters=500, lr=5e-2,
                 use_scheduler=False, track_ls=True, track_hyper_trace=True, fix_outputscale=False, save_path=None):
        self.model = model
        self.likelihood = likelihood
        #self.iterator = tqdm(range(total_iters), desc="{} GP Training".format(method_name))
        self.total_iters = total_iters
        self.train_x, self.train_y = model.train_inputs[0], model.train_targets
        if fix_outputscale:
            params = [model.covar_module.base_kernel.raw_lengthscale, model.likelihood.noise_covar.raw_noise]
        else:
            params = model.parameters()
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = \
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=self.optimizer,
                    milestones=[int(0.5*total_iters), int(0.7*total_iters), int(0.9*total_iters)],
                    gamma=0.1)

        self.mll = mll
        self.loss, self.ls, self.os, self.noise = 0., 0., 0., 0.
        self.track_hyper_trace = track_hyper_trace
        self.track_ls = track_ls
        if self.track_hyper_trace:
            if track_ls:
                # TODO: change it to accomodating ard kernels
                self.hyper_trace = {'ls': np.zeros(total_iters),
                                'os': np.zeros(total_iters),
                                'noise': np.zeros(total_iters),
                                'loss': np.zeros(total_iters)}
            else:
                self.hyper_trace = {#ls': np.zeros(total_iters, ard_dim),
                                'os': np.zeros(total_iters),
                                'noise': np.zeros(total_iters),
                                'loss': np.zeros(total_iters)}
        self.iter = 0
        self.dist_of_iter = dist_of_iter

        self.logger = logger
        assert method in ['cholesky', 'rrcg', 'cg', 'gpt-cholesky', 'gpt-cg'], method
        self.method = method
        self.save_path = save_path

    def _train_gpt(self):
        """this is for gpytorch model which does not receive dist_of_iter as input"""
        self.model.train()
        #for _ in self.iterator:
        for _ in range(self.total_iters):
            tic = time.time()
            self._take_one_step_gpt()
            toc = time.time()
            self.update_trackers(time_taken=toc-tic)

    def _take_one_step_gpt(self):
        self.optimizer.zero_grad()
        y_hat = self.model(self.train_x)
        self.loss = -self.mll(y_hat, self.train_y)
        self.loss.backward()
        self.optimizer.step()
        if self.use_scheduler:
            self.scheduler.step()

    def train(self):
        if self.method == 'gpt-cholesky' or self.method == 'gpt-cg':
            self._train_gpt()
            return

        self.model.train()
        #for _ in self.iterator:
        for itr in range(self.total_iters):
            tic = time.time()
            self.take_one_step()
            toc = time.time()
            self.update_trackers(time_taken=toc-tic)

            if (itr + 1) % 50 == 0:
                if self.save_path is not None:
                    save_path = self.save_path + f".iter{itr + 1}"
                    print(f"Saving model at {save_path}")
                    torch.save(self.model.state_dict(), save_path)


    def take_one_step(self):
        self.optimizer.zero_grad()
        y_hat = self.model(self.train_x)
        self.loss = -self.mll(y_hat, self.train_y, dist_of_iter=self.dist_of_iter)
        self.loss.backward()
        self.optimizer.step()
        if self.use_scheduler:
            self.scheduler.step()

    def update_trackers(self, time_taken):
        self.update_params()
        #update_iterator_print(self.iterator, self.loss.item(),
        #                      self.ls, self.os, self.noise, lr=self.optimizer.param_groups[0]['lr'])
        update_logger(self.logger, loss=self.loss,  lr=self.optimizer.param_groups[0]['lr'], iteration=self.iter,
                      ls=self.ls, os=self.os, noise=self.noise,
                      time_taken=time_taken, track_ls=self.track_ls)
        if self.track_hyper_trace:
            self.update_results()
        self.iter += 1

    def update_params(self):
        #outputscale = 0.62923026
        #self.model.covar_module.outputscale = outputscale
        if self.track_ls:
            # if ard kernel, then lengthscale is of shape (1, ard_num_dim)
            self.ls = self.model.covar_module.base_kernel.lengthscale.item()
        else:
            self.ls = self.model.covar_module.base_kernel.lengthscale.squeeze().tolist()
        self.os = self.model.covar_module.outputscale.item()
        self.noise = self.likelihood.noise.item()

    def update_results(self):
        if self.track_ls:
            self.hyper_trace['ls'][self.iter] = self.ls
        self.hyper_trace['os'][self.iter] = self.os
        self.hyper_trace['noise'][self.iter] = self.noise
        self.hyper_trace['loss'][self.iter] = self.loss


def update_logger(logger, loss, lr, ls, os, noise, iteration, time_taken, track_ls=True):
    if track_ls:
        text = f'iter: {iteration:4d} | loss: {loss:+1.4e} | lr: {lr:.5f} | '
        text += f'ls: {ls:4.6f} | noise: {noise:4.6f} | os: {os:4.6f} | '
        text += f'time: {time_taken:4.2f} sec'
    else:
        text = f'iter: {iteration:4d} | loss: {loss:+1.4e} | lr: {lr:.5f} | '
        text += 'ls: ' + ','.join([f'{sub_ls:4.6f}' for sub_ls in ls]) + ' | '
        text += f'noise: {noise:4.6f} | os: {os:4.6f} | '
        text += f'time: {time_taken:4.2f} sec'
    logger.info(text)


def start_all_logging_instruments(settings, results_path, time_stamp=None):
    if time_stamp is None:
        time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_file_name = results_path + 'loss_' + time_stamp + '.log'
    logger_name = 'log_' + time_stamp
    logger = setup_logger(log_file_name, logger_name)
    log_all_settings(settings, logger)
    return logger


def log_all_settings(settings, logger):
    for key, value in settings.items():
        logger.info(f'Hyper: {key}: {value}')


def setup_logger(log_file_name, logger_name: str = None):
    if logger_name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:    %(message)s')
    stream_formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(filename=log_file_name)
    file_handler.setFormatter(fmt=formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=stream_formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=stream_handler)
    logger.propagate = False
    return logger


def print_dist_stats(dist):
    print("mean = ", dist.mean)
    print("std = ", dist.std)


def get_dist_of_iter(**dist_of_iter_kwargs):
    dist_type = dist_of_iter_kwargs['rr_dist_type']
    assert dist_type in ['expdecay', 'oneoverj', 'geometric'], \
        "dist_type must be among expdecay, oneoverj, geometric, but got {}".format(dist_type)

    min = dist_of_iter_kwargs.get("rr_iter_min", 1)
    max = dist_of_iter_kwargs['rr_iter_max']
    default_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = dist_of_iter_kwargs.get("device", default_device)
    dtype = dist_of_iter_kwargs.get("dtype", torch.float)
    if dist_type == 'expdecay':
        temp = dist_of_iter_kwargs['temp']
        dist_of_iter = ExpDecayDist(temp=temp, min=min, max=max, dtype=dtype, device=device)
    elif dist_type == 'oneoverj':
        do_sqrt = dist_of_iter_kwargs.get("do_sqrt", False)
        dist_of_iter = OneOverJ(min=min, max=max, do_sqrt=do_sqrt, dtype=dtype, device=device)
    else:
        p = dist_of_iter_kwargs.get("p", 0.5)
        dist_of_iter = Geometric(p=p, min=min, max=max, dtype=dtype, device=device)
    return dist_of_iter


def print_initial_hypers(model, print_ls=True):
    noise_scale = model.likelihood.noise_covar.noise.item()
    output_scale = model.covar_module.outputscale.item()

    if print_ls:
        ls = model.covar_module.base_kernel.lengthscale.item()
        print(f'Pre training: noise scale {noise_scale: 4.4f} | ' +
              f'lengthscale {ls:4.4f} | ' +
              f'output scale {output_scale:4.4f}')
    else:
        print(f'Pre training: noise scale {noise_scale: 4.4f} | ' +
              f'output scale {output_scale:4.4f}')


def set_hypers(model, noise_scale, ls, output_scale):
    hypers = {
        'likelihood.noise_covar.noise': noise_scale,
        'covar_module.base_kernel.lengthscale': ls,
        'covar_module.outputscale': output_scale,
    }
    model.initialize(**hypers)


def get_hypers(model):
    hypers = {}
    hypers["noise_scale"] = model.likelihood.noise_covar.noise.cpu().item()
    hypers["output_scale"] = model.covar_module.outputscale.cpu().item()
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    is_using_ard = ls.shape[1] > 1
    if is_using_ard:
        for i in range(ls.shape[1]):
            key = 'ls_' + str(i)
            hypers[key] = ls[0, i]
    else:
        hypers["ls"] = ls.item()
    return hypers


def get_string_time_taken(tic, toc, experiment_name="Experiment"):
    minutes = round((toc - tic) / 60)
    seconds = (toc - tic) - minutes * 60
    return f'{experiment_name} took: {minutes:4d} min and {seconds:4.2f} sec'


def save_results(results, output_file):
    with open(file=output_file, mode='wb') as f:
        pickle.dump(obj=results, file=f)


class GPRegressionModel(gpt.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type='rbf', use_keops=False, ard_dim=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpt.means.ZeroMean()
        if torch.cuda.is_available() and use_keops:
            if kernel_type == 'rbf':
                kernel = gpt.kernels.keops.RBFKernel()
            elif kernel_type == 'matern05':
                kernel = gpt.kernels.keops.MaternKernel(nu=0.5)
            elif kernel_type == 'matern15':
                kernel = gpt.kernels.keops.MaternKernel(nu=1.5)
            elif kernel_type == 'matern25':
                kernel = gpt.kernels.keops.MaternKernel(nu=2.5)
            elif kernel_type == 'rbf-ard':
                if ard_dim is None:
                    ard_dim = train_x.shape[-1]
                    print(f"Getting ard_dim from training input, ard_dim = {ard_dim}")
                    kernel = gpt.kernels.keops.RBFKernel(ard_num_dims=ard_dim)
            else:
                raise ValueError("kernel_type must be chosen among {}, but got {}.".format(
                    ['rbf', 'matern05', 'matern15', 'matern25'], kernel_type))
        else:
            if kernel_type == 'rbf':
                kernel = gpt.kernels.RBFKernel()
            elif kernel_type == 'matern05':
                kernel = gpt.kernels.MaternKernel(nu=0.5)
            elif kernel_type == 'matern15':
                kernel = gpt.kernels.MaternKernel(nu=1.5)
            elif kernel_type == 'matern25':
                kernel = gpt.kernels.MaternKernel(nu=2.5)
            elif kernel_type == 'rbf-ard':
                if ard_dim is None:
                    ard_dim = train_x.shape[-1]
                    print(f"Getting ard_dim from training input, ard_dim = {ard_dim}")
                    kernel = gpt.kernels.RBFKernel(ard_num_dims=ard_dim)
            else:
                raise ValueError("kernel_type must be chosen among {}, but got {}.".format(
                    ['rbf', 'matern05', 'matern15', 'matern25'], kernel_type))

        self.covar_module = gpt.kernels.ScaleKernel(
            kernel,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)


