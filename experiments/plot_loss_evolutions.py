import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from experiments.plot_fns import increase_size_of_vect
from experiments.plot_fns import get_colors_dict
from experiments.plot_fns import smooth_vec
from experiments.utils import load_results

sns.set_theme(style='darkgrid')
save_output = False
results = load_results('./results/saved/all_results_exact.pkl')
max_iter = []

for key in results.keys():
    max_iter.append(len(results[key]['loss']))
max_iter = np.max(max_iter)
ccc = get_colors_dict()
variables = ['loss', 'exact_loss']

info = {
    'cholesky': {'label': 'Cholesky', 'color': ccc['cholesky'], 'style': 'dashed'},
    'rff1': {'label': r'RFF($J=200$)', 'color': ccc['rff1'], 'style': '-'},
    'rff3': {'label': r'RFF($J=700$)', 'color': ccc['rff'], 'style': '-'},
    'rff2': {'label': r'RFF($J=500$)', 'color': ccc['rff3'], 'style': '-'},
    'ssrff': {'label': r'SS-RFF($E[J=700]$)', 'color': ccc['ssrff'], 'style': '-'},
    'cg1': {'label': r'CG(20)', 'color': ccc['cg2'], 'style': '-'},
    'cg2': {'label': r'CG(40)', 'color': ccc['cg'], 'style': 'dashed'},
    'cg': {'label': r'CG(60)', 'color': ccc['cg'], 'style': '-'},
    'rrcg': {'label': 'RR-CG', 'color': ccc['rrcg'], 'style': '-'},
    'chol_cg': {'label': 'Cholesky', 'color': ccc['cholesky'], 'style': 'dashed'},
}

all_cases = [
    ('ssrff', ['rff1', 'rff2', 'rff3', 'cholesky']),
    ('cg1', ['cg2', 'rrcg'])
]
for key, models in all_cases:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    var = variables[1]
    vec = results[key][var]
    ax.set_title('Exact loss', fontsize=18)
    ax.plot(smooth_vec(vec, std=1.0, num=25), color=info[key]['color'],
            label=info[key]['label'])
    for model in models:
        if key == 'ssrff':
            vec = increase_size_of_vect(results[model][var], max_iter)
        else:
            vec = results[model][var]
        vec = smooth_vec(vec, std=1.0, num=25)
        ax.plot(vec, color=info[model]['color'], label=info[model]['label'],
                linestyle=info[model]['style'], linewidth=2.)
    ax.set_xscale('log')
    ax.set_xlabel('# of iterations (log-scale)', fontsize=18)
    ax.set_ylabel(r'$-\log \, p(y|X)$', fontsize=18)
    ax.legend(fontsize=7)

    plt.tight_layout()
    if save_output:
        plt.savefig('./results/loss_evolution_' + key + '.png')
        plt.savefig('./paper/figures/loss_evolution_' + key + '.png')
        plt.savefig('./results/loss_evolution_' + key + '.pdf')
        plt.savefig('./paper/figures/loss_evolution_' + key + '.pdf')
    plt.show()
