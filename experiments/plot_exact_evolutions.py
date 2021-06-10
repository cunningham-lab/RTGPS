# import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from experiments.plot_fns import increase_size_of_vect
from experiments.plot_fns import get_colors_dict
from experiments.plot_fns import smooth_vec
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

all_cases = [
    ('rff1', ['rff2', 'ssrff', 'ssrff2', 'cholesky']),
    # ('cg1', ['cg2', 'rrcg', 'rrcg2', 'chol_cg'])
    ('cg1', ['rrcg', 'chol_cg'])
]
xlabel = '# of optimization steps (log-scale)'
for key, models in all_cases:
    # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    var = variables[1]
    vec = results[key][var]
    ax.set_title('Negative log marginal likelihood', fontsize=18)
    vec = increase_size_of_vect(results[key][var], 1000)
    ax.plot(smooth_vec(vec, std=1.0, num=25), color=info[key]['color'],
            label=info[key]['label'])
    for model in models:
        if key == 'rff2':
            vec = increase_size_of_vect(results[model][var], max_iter)
        else:
            vec = results[model][var]
            vec = increase_size_of_vect(results[model][var], max_iter)
        vec = smooth_vec(vec, std=1.0, num=25)
        ax.plot(vec, color=info[model]['color'], label=info[model]['label'],
                linestyle=info[model]['style'], linewidth=2.)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(r'$-\log \, p(y|X,\theta)$', fontsize=18)
    # ax.set_ylim([0, 7])
    # if key == 'cg1':
    #     ax.set_ylim([0.1, 0.8])
    ax.legend(fontsize=12)

    plt.tight_layout()
    if save_output:
        # plt.savefig('./results/exact_evolution_' + key + '.png')
        # plt.savefig('./paper/figures/exact_evolution_' + key + '.png')
        plt.savefig('./results/exact_evolution_bike_' + key + '.pdf')
        # plt.savefig('./paper/figures/exact_evolution_' + key + '.pdf')
    plt.show()
