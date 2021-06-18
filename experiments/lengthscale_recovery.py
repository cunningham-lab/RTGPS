import pickle
import math
import time
import numpy as np
import torch
import gpytorch as gpt
from experiments.experiment_fns import GPRegressionModel
from experiments.experiment_fns import set_hypers
from experiments.experiment_fns import recover_rff
from experiments.experiment_fns import recover_cg
from experiments.experiment_fns import sample_from_prior

train_n, test_n, lengthscale_num, cg_max, use_all_rff = 10, 5, 1, 5, False
# train_n, test_n, lengthscale_num, cg_max, use_all_rff = 200, 151, 11, 54, True
save_results = True
use_cuda = torch.cuda.is_available()
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
output_file = './results/ls_recovery.pkl'

rff_samples = np.concatenate((np.array([50]),
                              np.arange(100, 1000, 100),
                              np.arange(1000, 3000, 250)))
rff_samples = rff_samples if use_all_rff else np.array([10])
cg_iters = np.arange(4, cg_max, step=4)
lower, upper = 0.1, 1.0
true_lengthscales = torch.tensor(np.linspace(lower, upper, lengthscale_num))

output_scale = torch.tensor(2.0)
noise_scale = torch.tensor(0.01)
train_x = torch.linspace(-3, 3, train_n)
train_y = torch.sin(train_x * (5 * math.pi)) * train_x + torch.randn_like(train_x) * 0.1
test_x = torch.linspace(-4, 5, test_n)
train_x = train_x.cuda() if use_cuda else train_x
train_y = train_y.cuda() if use_cuda else train_y
test_x = test_x.cuda() if use_cuda else test_x

recovered_ls_rff = np.zeros((lengthscale_num, rff_samples.shape[0]))
recovered_ls_cg = np.zeros((lengthscale_num, cg_iters.shape[0]))
training_time_rff = np.zeros_like(recovered_ls_rff)
training_time_cg = np.zeros_like(recovered_ls_cg)

init_lenghtscales_rff = (lower - upper) * torch.rand(lengthscale_num) + upper
init_lenghtscales_cg = (lower - upper) * torch.rand(lengthscale_num) + upper

print('Experimental params:')
print('Lengthscales:')
print(true_lengthscales)
print('RFF samples:')
print(rff_samples)
print('CG_iters:')
print(cg_iters)

t0 = time.time()
for i in range(len(true_lengthscales)):
    true_likelihood = gpt.likelihoods.GaussianLikelihood()
    true_likelihood = true_likelihood.cuda() if use_cuda else true_likelihood
    true_model = GPRegressionModel(train_x, train_y, true_likelihood)
    true_model = true_model.cuda() if use_cuda else true_model
    set_hypers(true_model, noise_scale, torch.tensor(true_lengthscales[i]), output_scale)
    true_ls = true_model.covar_module.base_kernel.lengthscale.item()
    print(f'True lengthscale = {true_ls:2.2f}')
    train_y = sample_from_prior(true_model, true_likelihood, train_x)
    train_ds = (train_x, train_y)

    for j, rff_sample in enumerate(rff_samples):
        tic = time.time()
        print(f'\nTrue lengthscale = {true_ls:2.2f}')
        hyperparams = (noise_scale, init_lenghtscales_rff[i], output_scale)
        model, ls, *_ = recover_rff(rff_sample, hyperparams, train_ds)
        recovered_ls_rff[i, j] = float(ls)
        training_time_rff[i, j] = time.time() - tic

    for c, cg_iter in enumerate(cg_iters):
        tic = time.time()
        print(f'\nTrue lengthscale = {true_ls:2.2f}')
        hyperparams = (noise_scale, init_lenghtscales_cg[i], output_scale)
        cg_model, cg_ls, *_ = recover_cg(cg_iter, hyperparams, train_ds)
        recovered_ls_cg[i, c] = float(cg_ls)
        training_time_cg[i, c] = time.time() - tic

t1 = time.time()
print(f'Experiment took: {t1 - t0:4.2f} sec')

if save_results:
    output = {'RFF': {'ls': recovered_ls_rff, 'time': training_time_rff},
              'CG': {'ls': recovered_ls_cg, 'time': training_time_cg},
              'True': {'ls': true_lengthscales},
              'Conditions': {'rff_samples': rff_samples, 'cg_iters': cg_iters}}
    with open(file=output_file, mode='wb') as f:
        pickle.dump(obj=output, file=f)
