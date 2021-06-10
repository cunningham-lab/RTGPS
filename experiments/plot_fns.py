import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import torch
import gpytorch as gpt
from experiments.experiment_fns import GPRegressionModel, set_hypers
from experiments.load_data import load_uci_data
from experiments.utils import load_results


def append_unbiasedness_cg_results(results, input_file):
    res = load_results(input_file)
    for k, output in res.items():
        if k == 'cholesky':
            logdet = output['logdet'].item()
            invquad = output['inv_quad'].item()
            # logdet = output['logdet']
            # invquad = output['inv_quad']
            results['chol_cg'] = (logdet, invquad)
        else:
            rounds = []
            for i in output['logdet'].keys():
                track = {}
                for v in ['logdet', 'inv_quad']:
                    aux = 'invquad' if v == 'inv_quad' else v
                    # track.update({aux: output[v][i]})
                    track.update({aux: np.array(output[v][i])})
                rounds.append(track)
            results[k] = rounds
    results['num_cg_iters'] = make_keys_to_int(res['cg']['logdet'].keys())
    return results


def make_keys_to_int(keys):
    output = []
    for idx in keys:
        output.append(int(idx))
    output = np.array(output, dtype=np.int32)
    return output


def get_exact_loss_from_hypers_logged(ls, noise, os, run_sample, dataset_name='pol'):
    train_x, train_y, *_ = load_uci_data('./datasets/', dataset_name)
    if run_sample:
        train_n = int(1.e2)
        train_x, train_y = train_x[:train_n, :], train_y[:train_n]

    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    mll = gpt.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()

    # with gpt.settings.max_cholesky_size(int(1.0e7)):
    # gpt.settings.max_cholesky_size(int(1.0e7))
    loss = torch.zeros(len(os))

    for idx, _ in enumerate(os):
        model.train()
        set_hypers(model, noise[idx], ls[idx], os[idx])
        loss[idx] = -mll(model(train_x), train_y).item()
        print(f"Going over observation: {idx:4d}")
    return loss


def smooth_vec(v, std, num):
    df = pd.DataFrame(v)
    rolling = np.array(df.rolling(num, win_type='gaussian', center=True).mean(std=std))
    return rolling


def increase_size_of_vect(v, max_iter):
    if len(v) < max_iter:
        diff = max_iter - len(v)
        a1 = v
        a2 = v[-1] * np.ones(diff)
        v = np.concatenate((a1, a2))
    return v


def get_gray_colormap():
    total = 10
    colors = [cm.Greys(x) for x in np.linspace(0.4, 0.9, total)]
    colors[0] = cm.Greys(0.2)
    grey_map = LinearSegmentedColormap.from_list('ggg', colors, N=total)
    return grey_map


def get_colors_dict():
    ccc = {
        'cholesky': '#cb181d',
        'rff1': '#c6dbef',
        'rff2': '#9ecae1',
        'rff3': '#6baed6',
        'rff': '#3182bd',
        'ssrff': '#08519c',
        'cg1': '#dadaeb',
        'cg2': '#bcbddc',
        'cg3': '#9e9ac8',
        'cg': '#756bb1',
        'rrcg': '#54278f'
    }
    return ccc
