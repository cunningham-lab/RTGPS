import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from experiments.utils import load_results
from experiments.plot_fns import get_colors_dict
from experiments.plot_fns import append_unbiasedness_cg_results

sns.set_theme(style='darkgrid')
ccc = get_colors_dict()
save_output = True
# save_output = False
input_file = './results/saved/unbiased_rff_100.pkl'
results = load_results(input_file)

input_file = './rrcg_experiments/exp-output-unbiasedness/'
input_file += 'pol_seed10_pol_ntrain9600_kernelrbf_rrnum1000_numprobes10.pkl'
input_file = './results/saved/10k_pol.pkl'
results = append_unbiasedness_cg_results(results, input_file)

log_det_exact, inv_quad_exact = results['cholesky']
num_rff_samples = results['num_samples']
# num_cg_iters = results['num_cg_iters']
num_cg_iters = results['num_cg_iters'][:-1]
num_rr_rounds = results['rff'][0]['J'].shape[0]
decimal = (100, 100)
decimal_cg = (22000, 6100)
log_det_exact /= decimal[0]
inv_quad_exact /= decimal[1]
order = num_rff_samples.argsort()
keys = ['rff', 'ssrff', 'rrcg', 'cg']
cases = ['logdet', 'invquad']
plotting = {}

for key in keys:
    total_rounds = len(results['rff']) if key in ['rff', 'ssrff'] else 9
    mean_std = {}
    for case in cases:
        mean_std.update({case: {'mean': np.zeros(total_rounds),
                                'std': np.zeros(total_rounds)}})
    plotting.update({key: mean_std})

for key in keys:
    total_rounds = len(results['rff']) if key in ['rff', 'ssrff'] else 9
    auxk = decimal_cg if key in ['rrcg', 'cg'] else decimal
    for case in cases:
        for i in range(total_rounds):
            res = results[key][order[i]][case]
            aux = auxk[0] if case == 'logdet' else auxk[1]
            plotting[key][case]['mean'][i] = np.mean(res / aux)
            plotting[key][case]['std'][i] = np.std(res / aux)


plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.suptitle('RFF', fontsize=18, x=0.525, y=0.95)
plt.hlines(y=log_det_exact, xmin=np.min(num_rff_samples),
           xmax=np.max(num_rff_samples),
           linestyle='dashed', color=ccc['cholesky'],
           label='Cholesky', linewidth=2.)
plt.plot(num_rff_samples[order], plotting['rff']['logdet']['mean'],
         '-o',
         label='RFF',
         linewidth=2.,
         linestyle='dashdot',
         color=ccc['rff'])
sample_size = results['ssrff'][0]['logdet'].shape[0]
plt.errorbar(num_rff_samples[order], plotting['ssrff']['logdet']['mean'],
             plotting['ssrff']['logdet']['std'] / np.sqrt(sample_size),
             label='SS-RFF',
             linewidth=2.,
             color=ccc['ssrff'])
plt.xlabel('# of RFF features', fontsize=18)
plt.ylabel(r'$\log|\hat{K}_{XX}|$', fontsize=18)
# plt.xticks(np.arange(200, 560, 50))
plt.xticks(np.arange(200, 600, 100))
plt.legend()

plt.subplot(122)
plt.hlines(y=inv_quad_exact, xmin=np.min(num_rff_samples[order]),
           xmax=np.max(num_rff_samples[order]),
           linewidth=2.,
           linestyle='dashed', color=ccc['cholesky'], label='Cholesky')
plt.plot(num_rff_samples[order],
         plotting['rff']['invquad']['mean'],
         '-o',
         label='RFF',
         linewidth=2.,
         linestyle='dashdot',
         color=ccc['rff'])
plt.errorbar(num_rff_samples[order],
             plotting['ssrff']['invquad']['mean'],
             plotting['ssrff']['invquad']['std'] / np.sqrt(sample_size),
             label='SS-RFF',
             linewidth=2.,
             color=ccc['ssrff'])
plt.xlabel(r'# of RFF features', fontsize=18)
plt.ylabel(r'$y^{T} \hat{K}_{XX}^{-1} y$', fontsize=18)
plt.legend()
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%4.1f'))
# plt.xticks(np.arange(200, 560, 50))
plt.xticks(np.arange(200, 600, 100))
plt.tight_layout()

if save_output:
    plt.savefig('./results/unbiasedness_rff.png')
    plt.savefig('./paper/figures/unbiasedness_rff.png')
    plt.savefig('./results/unbiasedness_rff.pdf')
    plt.savefig('./paper/figures/unbiasedness_rff.pdf')
plt.show()

plt.figure(figsize=(8, 4))
plt.suptitle('CG', fontsize=18, x=0.525, y=0.95)
plt.subplot(121)
log_det_exact = results['chol_cg'][0]
log_det_exact /= decimal_cg[0]
plt.hlines(y=log_det_exact, xmin=np.min(num_cg_iters),
           xmax=np.max(num_cg_iters),
           linewidth=2.,
           linestyle='dashed', color=ccc['cholesky'], label='Cholesky')
sample_size = results['rrcg'][0]['logdet'].shape[0]
plt.plot(num_cg_iters,
         plotting['cg']['logdet']['mean'],
         '-o',
         label='CG',
         linewidth=2.,
         linestyle='dashdot',
         color=ccc['cg'])
plt.errorbar(num_cg_iters,
             plotting['rrcg']['logdet']['mean'],
             plotting['rrcg']['logdet']['std'] / np.sqrt(sample_size),
             label='RR-CG',
             linewidth=2.,
             color=ccc['rrcg'])
plt.xlabel(r'# of CG iters', fontsize=18)
plt.ylabel(r'$\log|\hat{K}_{XX}|$', fontsize=18)
plt.legend()

plt.subplot(122)
inv_quad_exact = results['chol_cg'][1]
inv_quad_exact /= decimal_cg[1]
plt.hlines(y=inv_quad_exact, xmin=np.min(num_cg_iters),
           xmax=np.max(num_cg_iters),
           linewidth=2.,
           linestyle='dashed',
           color=ccc['cholesky'],
           label='Cholesky')
plt.plot(num_cg_iters,
         plotting['cg']['invquad']['mean'],
         '-o',
         linewidth=2.,
         linestyle='dashdot',
         label='CG',
         color=ccc['cg'])
plt.errorbar(num_cg_iters,
             plotting['rrcg']['invquad']['mean'],
             plotting['rrcg']['invquad']['std'] / np.sqrt(sample_size),
             linewidth=2.,
             label='RR-CG',
             color=ccc['rrcg'])
plt.xlabel(r'# of CG iters', fontsize=18)
plt.ylabel(r'$y^{T} \hat{K}_{XX}^{-1} y$', fontsize=18)
plt.legend()
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%4.1f'))
plt.tight_layout()

if save_output:
    plt.savefig('./results/unbiasedness_cg.png')
    plt.savefig('./paper/figures/unbiasedness_cg.png')
    plt.savefig('./results/unbiasedness_cg.pdf')
    plt.savefig('./paper/figures/unbiasedness_cg.pdf')
plt.show()
