import pickle
import numpy as np
from matplotlib import pyplot as plt
from experiments.experiment_fns import compute_recovered_vs_true_matrix
from experiments.experiment_fns import prepare_ax
from experiments.experiment_fns import get_lower_upper_via_quantiles

input_file = './results/mse_complexity_2020_11_09_16_53_18.pkl'

with open(input_file, mode='rb') as f:
    results = pickle.load(f)
rff_samples = results['rff_samples']
cg_iters = results['cg_iters']

div_error_rff = np.log(compute_recovered_vs_true_matrix(results['rff']['inv_quad'],
                                                        results['chol']['inv_quad']))
div_error_cg = np.log(compute_recovered_vs_true_matrix(results['cg']['inv_quad'],
                                                       results['chol']['inv_quad']))

fig, ax = plt.subplots(1, 2, figsize=(14, 8))
desc = {'title': 'RFF', 'xlabel': 'N RFF samples', 'ylabel': 'log(rff / cholesky)'}
lower, upper = get_lower_upper_via_quantiles(div_error_rff)
# lower, upper = -0.5, 0.5
ax[0] = prepare_ax(rff_samples, div_error_rff, ax[0], desc, [lower, upper])

desc = {'title': 'CG', 'xlabel': 'N CG iters', 'ylabel': 'log(cg / cholesy)'}
lower, upper = get_lower_upper_via_quantiles(div_error_cg)
# lower, upper = -0.5, 0.5
ax[1] = prepare_ax(cg_iters, div_error_cg, ax[1], desc, [lower, upper])
fig.suptitle(r'Comparing $y^{T} K^{-1} y$')
plt.savefig('./results/div_inv_quad.png')

div_error_rff = np.log(compute_recovered_vs_true_matrix(results['rff']['logdet'],
                                                        results['chol']['logdet']))
div_error_cg = np.log(compute_recovered_vs_true_matrix(results['cg']['logdet'],
                                                       results['chol']['logdet']))
fig, ax = plt.subplots(1, 2, figsize=(14, 8))
desc = {'title': 'RFF', 'xlabel': 'N RFF samples', 'ylabel': 'log(rff / cholesky)'}
# lower, upper = -0.5, 0.5
lower, upper = get_lower_upper_via_quantiles(div_error_rff)
ax[0] = prepare_ax(rff_samples, div_error_rff, ax[0], desc, [lower, upper])

desc = {'title': 'CG', 'xlabel': 'N CG iters', 'ylabel': 'log(cg / cholesky)'}
# lower, upper = -0.5, 0.5
lower, upper = get_lower_upper_via_quantiles(div_error_cg)
ax[1] = prepare_ax(cg_iters, div_error_cg, ax[1], desc, [lower, upper])

fig.suptitle(r'Comparing $\log(|K|)$')
plt.savefig('./results/div_logdet.png')
plt.show()
