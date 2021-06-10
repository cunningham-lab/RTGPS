import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from experiments.plot_fns import get_colors_dict
from experiments.utils import load_results

sns.set_theme(style='darkgrid')
save_output = True
results = load_results('./results/all_results_exact_bike.pkl')
max_iter = 1000
ccc = get_colors_dict()
variables = ['loss', 'exact_loss']

info = {
    'cholesky': {'label': 'Cholesky', 'color': ccc['cholesky'], 'style': 'dashed'},
    'rff1': {'label': r'RFF($J=1500$)', 'color': ccc['ssrff'], 'style': '-'},
    'rff2': {'label': r'RFF($J=1000$)', 'color': ccc['rff3'], 'style': '-'},
    'ssrff': {'label': r'SS-RFF($E[J]=1000$)', 'color': ccc['ssrff'], 'style': 'dashed'},
    'ssrff2': {'label': r'SS-RFF($E[J]=1500$)', 'color': ccc['rff3'], 'style': 'dashed'},
    'cg1': {'label': r'CG($J=20$)', 'color': ccc['cg'], 'style': 'dashed'},
    'cg2': {'label': r'CG($J=40$)', 'color': ccc['cg'], 'style': '-'},
    'cg': {'label': r'CG($J=60$)', 'color': ccc['cg'], 'style': '-'},
    'rrcg': {'label': 'RR-CG($E[J]=20$)', 'color': ccc['cg'], 'style': 'dashed'},
    'rrcg2': {'label': 'RR-CG($E[J]=40$)', 'color': ccc['rrcg'], 'style': 'dashed'},
    'chol_cg': {'label': 'Cholesky', 'color': ccc['cholesky'], 'style': 'dashed'},
}

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
var = variables[1]
key = 'ssrff'
ax.set_title('Negative log marginal likelihood', fontsize=18)

# vec = results[key][var]
vec = results[key][var][700:800]
# ax.plot(smooth_vec(vec, std=1.0, num=25), color=info[key]['color'],
#         label=info[key]['label'])
ax.plot(np.arange(700, 800), vec, color=info[key]['color'], label=info[key]['label'])
# ax.plot(np.arange(700, 800), 0.72 * np.ones(100),
#         color=info['cholesky']['color'], label=info['cholesky']['label'])

# vec = results['cholesky'][var]
# ax.plot(vec, color=info['cholesky']['color'], label=info['cholesky']['label'])
xlabel = '# of optimization steps'
ax.set_xlabel(xlabel, fontsize=18)
ax.set_ylabel(r'$-\log \, p(y|X,\theta)$', fontsize=18)
ax.legend(fontsize=12)

plt.tight_layout()
if save_output:
    plt.savefig('./results/stochasticity.pdf')
plt.show()

