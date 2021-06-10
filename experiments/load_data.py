import logging
import os
import math
from math import floor
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
import gpytorch as gpt
import git


def load_uci_data(data_dir, dataset, total_n=-1, train_p=0.64, val_p=0.16, test_p=0.20, cuda=False, verbose=False):
    """By default, use 64/16/20 for train/val/test split"""

    assert (train_p + val_p + test_p == 1) and (0<=train_p<=1) and (0<=test_p<=1), \
        "train_p = {}, val_p = {}, test_p = {}".format(train_p, val_p, test_p)

    # file_path = os.path.join(data_dir, dataset, dataset + '.mat')
    if data_dir is None:
        repo = git.Repo('.', search_parent_directories=True)
        repo_dir = repo.working_tree_dir
        data_dir = os.path.join(repo_dir, 'experiments/datasets')

    file_path = os.path.join(data_dir, dataset + '.mat')
    data = torch.tensor(loadmat(file_path)['data'])
    data = data.float()
    X = data[:, :-1]
    y = data[:, -1]

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        no_var_dim = X.size(1) - int(good_dimensions.sum())
        logging.info(f"Removed {no_var_dim:d} dimensions with no variance")
        X = X[:, good_dimensions]

    if dataset in ['keggundirected', 'slice']:
        X = SimpleImputer(missing_values=np.nan).fit_transform(X.data.numpy())
        X = torch.Tensor(X)

    if verbose:
        print("Loading dataset {}...".format(dataset))
        print("X shape = {}\n\n".format(X.shape))

    # Shuffling, subsampling and normalizing
    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    if total_n != -1:
        assert 0 < total_n <= len(X), "total_n should be in (0, {}), but got {}".format(len(X), total_n)
        X = X[:total_n]
        y = y[:total_n]

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y -= y.mean()
    y /= y.std()

    train_n = int(floor(train_p * X.size(0)))
    valid_n = int(floor(val_p * X.size(0)))

    if cuda:
        split = split_dataset_w_cuda(X, y, train_n, valid_n)
    else:
        split = split_dataset(X, y, train_n, valid_n)
    train_x, train_y, valid_x, valid_y, test_x, test_y = split
    logging.info(f"Loaded data with input dimension of {test_x.size(-1):d}")

    return train_x, train_y, test_x, test_y, valid_x, valid_y


def load_uci_data_ap(data_dir, dataset, use_cuda=False):
    file_path = os.path.join(data_dir, dataset + '.mat')
    data = torch.tensor(loadmat(file_path)['data'])
    data = data.float()
    X = data[:, :-1]
    y = data[:, -1]

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        no_var_dim = X.size(1) - int(good_dimensions.sum())
        logging.info(f"Removed {no_var_dim:d} dimensions with no variance")
        X = X[:, good_dimensions]

    if dataset in ['keggundirected', 'slice']:
        X = SimpleImputer(missing_values=np.nan).fit_transform(X.data.numpy())
        X = torch.Tensor(X)

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y -= y.mean()
    y /= y.std()

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    if use_cuda:
        split = split_dataset_w_cuda(X, y, train_n, valid_n)
    else:
        split = split_dataset(X, y, train_n, valid_n)
    train_x, train_y, valid_x, valid_y, test_x, test_y = split
    logging.info(f"Loaded data with input dimension of {test_x.size(-1):d}")

    return train_x, train_y, test_x, test_y, valid_x, valid_y


def get_train_data(data_dir, dataset_name, run_sample, sample_size, use_cuda):
    train_x, train_y, *_ = load_uci_data_ap(data_dir, dataset_name, use_cuda)
    if run_sample:
        train_n = sample_size
        train_x, train_y = train_x[:train_n, :], train_y[:train_n]
    train_ds = (train_x, train_y)
    return train_ds


def get_train_test_data(data_dir, dataset, run_sample=True, use_cuda=False):
    output = load_uci_data_ap(data_dir, dataset, use_cuda)
    train_x, train_y, test_x, test_y, valid_x, valid_y = output
    if run_sample:
        obs_n = int(1.e2)
        test_x, test_y = test_x[:obs_n, :], test_y[:obs_n]
        train_x, train_y = train_x[:obs_n, :], train_y[:obs_n]
        valid_x, valid_y = valid_x[:obs_n, :], valid_y[:obs_n]
    test_ds = (test_x, test_y)
    valid_ds = (valid_x, valid_y)
    train_ds = (train_x, train_y)
    return train_ds, test_ds, valid_ds


def split_dataset(x, y, train_n, valid_n):
    train_x = x[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    valid_x = x[train_n:train_n + valid_n, :].contiguous()
    valid_y = y[train_n:train_n + valid_n].contiguous()

    test_x = x[train_n + valid_n:, :].contiguous()
    test_y = y[train_n + valid_n:].contiguous()
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def split_dataset_w_cuda(x, y, train_n, valid_n):
    train_x = x[:train_n, :].contiguous().cuda()
    train_y = y[:train_n].contiguous().cuda()

    valid_x = x[train_n:train_n + valid_n, :].contiguous().cuda()
    valid_y = y[train_n:train_n + valid_n].contiguous().cuda()

    test_x = x[train_n + valid_n:, :].contiguous().cuda()
    test_y = y[train_n + valid_n:].contiguous().cuda()
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def sample_from_prior(model, likelihood, train_x):
    with gpt.settings.prior_mode(True):
        prior_preds = likelihood(model(train_x))
    return prior_preds.sample()
