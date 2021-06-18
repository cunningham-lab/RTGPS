import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from experiments.experiment_fns import compute_recovered_vs_true_matrix
from experiments.plot_fns import get_colors_dict
from experiments.utils import load_results

sns.set_theme(style='darkgrid')
save_output = True
input_file = './results/ls_recovery.pkl'
ccc = get_colors_dict()

results = load_results(input_file)
true_lengthscales = results['True']['ls']
recovered_ls_rff = results['RFF']['ls']
training_time_rff = results['RFF']['time']
recovered_ls_cg = results['CG']['ls']
training_time_cg = results['CG']['time']
rff_samples = results['Conditions']['rff_samples']
cg_iters = results['Conditions']['cg_iters']

div_error_rff = compute_recovered_vs_true_matrix(recovered_ls_rff, true_lengthscales)
div_error_cg = compute_recovered_vs_true_matrix(recovered_ls_cg, true_lengthscales)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].axhline(y=0.0, linestyle='dashed', color=ccc['cholesky'], linewidth=2.)
ax[0].plot(cg_iters, np.log(np.mean(div_error_cg, axis=0)), '-o',
           color=ccc['cg'], label='CG', linewidth=2., linestyle='dashdot')
ax[0].set_xticks(cg_iters)
ax[0].set_title('CG', fontsize=18)
ax[0].legend(fontsize=18)
ax[0].set_xlabel('# of CG iterations', fontsize=18, labelpad=25)
ax[0].set_ylabel(r'$\log(\hat{\ell^2}/\ell^2)$', fontsize=18)
ax[0].set_xticks(np.arange(0, 60, 5))
ax[1].axhline(y=0.0, linestyle='dashed',
              color=ccc['cholesky'], linewidth=2.)
ax[1].plot(rff_samples, np.log(np.mean(div_error_rff, axis=0)), '-o',
           color=ccc['rff'], label='RFF', linewidth=2., linestyle='dashdot')
ax[1].set_xticklabels(rff_samples[np.arange(0, len(rff_samples), 2)], rotation=60)
ax[1].set_title('RFF', fontsize=18)
ax[1].legend(fontsize=18)
ax[1].set_xlabel('# of RFF features', fontsize=18)
ax[1].set_ylabel(r'$\log(\hat{\ell^2}/{\ell^2})$', fontsize=18)
ax[1].set_xticks([50, 300, 500, 700, 1000, 1250, 1750, 2250, 2750])

fig.tight_layout()
if save_output:
    plt.savefig('./results/ls_recovery.png')
plt.show()
