import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from experiments.utils import load_results
from experiments.plot_fns import get_colors_dict
from experiments.plot_fns import get_gray_colormap

sns.set_theme(style='darkgrid')
save_output = True
input_grid_file = './results/saved/exact_loss_landscape_1000.csv'
grey_cmap = get_gray_colormap()

ccc = get_colors_dict()
exact_df = pd.read_csv(input_grid_file).drop("outputscale", axis=1)
exact_df = exact_df.set_index(["lengthscale", "noise"]).drop_duplicates().reset_index()
ls = exact_df["lengthscale"]
num_ls = len(ls.unique())
noise = exact_df["noise"]
num_noise = len(noise.unique())

ls_grid = ls.values.reshape((num_ls, num_noise))
noise_grid = noise.values.reshape((num_ls, num_noise))
mll = -exact_df["mll"].values.reshape((num_ls, num_noise))
results = {
    'cholesky': {'file': './results/saved/chol.pkl',
                 'div': 5
                 },
    'rff': {'file': './results/saved/rff.pkl',
            'div': 5
            },
    'ssrff': {'file': './results/saved/ssrff.pkl',
              'div': 10,
              'points': [i for i in range(0, 30, 3)] +
              [i for i in range(90, 100, 3)] +
              [i for i in range(100, 150, 3)] +
              [i for i in range(1000, 1200, 3)]
              },
    'cg': {'file': './results/saved/cg20.pkl',
           'div': 5
           },
    'rrcg': {'file': './results/saved/rrcg_20.pkl',
             'div': 5,
             }
}
for k, v in results.items():
    out = load_results(v['file'])
    # if k == 'ssrff':
    #     results[k]['ls'] = out['ls'][0, :]
    # else:
    #     results[k]['ls'] = out['ls']
    results[k]['ls'] = out['ls']
    results[k]['noise'] = out['noise']

xlabel = r'lengthscale ($\ell$)'
ylable = r'noise ($\sigma^{2}$)'
fontsize = 30

for k in results.keys():
    total = results[k]['ls'].shape[0]
    print(f'Model {k} has in total {total:3d} points')
    total = 100 if k == 'rrcg' else total
    points_ind = np.linspace(0, total - 1,
                             num=total / results[k]['div']).astype(np.int).tolist()
    points_ind = points_ind if not k == 'ssrff' else results[k]['points']
    num_points = len(points_ind)
    results[k]['x'] = np.zeros(num_points)
    results[k]['y'] = np.zeros(num_points)
    for i, points in enumerate(points_ind):
        results[k]['x'][i] = results[k]['ls'][points]
        results[k]['y'][i] = results[k]['noise'][points]

fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
fig.suptitle('Negative log marginal likelihood', fontsize=fontsize)
cf = ax[0].contourf(ls_grid, noise_grid, np.clip(mll, None, 1),
                    level=50, cmap=grey_cmap)
ax[0].set_xlabel(xlabel, fontsize=fontsize)
ax[0].set_ylabel(ylable, fontsize=fontsize)
ax[0].set_xlim((0.01, 1.))
ax[0].semilogy()
ax[0].set_ylim((0.03, 0.))
ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%4.1f'))
ax[0].yaxis.set_ticks([0.05, 0.1, 0.2, 0.4, 0.8])
ax[0].set_yticklabels([0.05, 0.1, 0.2, 0.4, 0.8])
ax[0].scatter(results['cholesky']['x'], results['cholesky']['y'],
              marker='o', color=ccc['cholesky'], label='Cholesky', s=(100,))
ax[0].scatter(results['cholesky']['x'][-1], results['cholesky']['y'][-1],
              marker='X', color='orange', s=(100,))
ax[0].legend(fontsize=20)

cf = ax[1].contourf(ls_grid, noise_grid, np.clip(mll, None, 1.),
                    cmap=grey_cmap)
ax[1].set_xlabel(xlabel, fontsize=fontsize)
ax[1].set_ylabel(ylable, fontsize=fontsize)
ax[1].set_xlim((0.01, 1.))
ax[1].semilogy()
ax[1].set_ylim((0.03, 0.))
ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%4.1f'))
ax[1].yaxis.set_ticks([0.05, 0.1, 0.2, 0.4, 0.8])
ax[1].set_yticklabels([0.05, 0.1, 0.2, 0.4, 0.8])
ax[1].scatter(results['rrcg']['x'], results['rrcg']['y'],
              marker='o', color=ccc['rrcg'], label='RR-CG', s=(100,))
ax[1].scatter(results['rrcg']['x'][-1], results['rrcg']['y'][-1],
              marker='X', color='orange', s=(100,))
ax[1].legend(fontsize=20)

cf = ax[2].contourf(ls_grid, noise_grid, np.clip(mll, None, 1.),
                    cmap=grey_cmap)
ax[2].set_xlabel(xlabel, fontsize=fontsize)
ax[2].set_ylabel(ylable, fontsize=fontsize)
ax[2].set_xlim((0.01, 1.))
ax[2].semilogy()
ax[2].set_ylim((0.03, 0.))
ax[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%4.1f'))
ax[2].yaxis.set_ticks([0.05, 0.1, 0.2, 0.4, 0.8])
ax[2].set_yticklabels([0.05, 0.1, 0.2, 0.4, 0.8])
ax[2].scatter(results['cg']['x'][:-86], results['cg']['y'][:-86],
              marker='o', color=ccc['cg'], label='CG', s=(100,))
ax[2].scatter(results['cg']['x'][-87], results['cg']['y'][-87],
              marker='X', color='orange', s=(100,))
ax[2].legend(fontsize=20)

if save_output:
    plt.savefig('./results/loss_landscapes_1.png')
    plt.savefig('./paper/figures/loss_landscapes_1.png')
    plt.savefig('./results/loss_landscapes_1.pdf')
    plt.savefig('./paper/figures/loss_landscapes_1.pdf')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
cf = ax[0].contourf(ls_grid, noise_grid, np.clip(mll, None, 1.),
                    level=50, cmap=grey_cmap)
ax[0].set_xlabel(xlabel, fontsize=fontsize)
ax[0].set_ylabel(ylable, fontsize=fontsize)
ax[0].set_xlim((0.01, 1.))
ax[0].semilogy()
ax[0].set_ylim((0.03, 0.))
ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%4.1f'))
ax[0].yaxis.set_ticks([0.05, 0.1, 0.2, 0.4, 0.8])
ax[0].set_yticklabels([0.05, 0.1, 0.2, 0.4, 0.8])
ax[0].scatter(results['ssrff']['x'], results['ssrff']['y'],
              marker='o', color=ccc['ssrff'], label='SS-RFF', s=(100,))
# ax[0].scatter(results['ssrff']['x'][-1], results['ssrff']['y'][-1],
#               marker='X', color='orange', s=(100,))
ax[0].scatter(results['ssrff']['x'][71], results['ssrff']['y'][71],
              marker='X', color='orange', s=(100,))
ax[0].legend(fontsize=20)


cf = ax[1].contourf(ls_grid, noise_grid, np.clip(mll, None, 1.),
                    level=50, cmap=grey_cmap)
ax[1].set_xlabel(xlabel, fontsize=fontsize)
ax[1].set_ylabel(ylable, fontsize=fontsize)
ax[1].set_xlim((0.01, 1.))
ax[1].semilogy()
ax[1].set_ylim((0.03, 0.))
ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%4.1f'))
ax[1].yaxis.set_ticks([0.05, 0.1, 0.2, 0.4, 0.8])
ax[1].set_yticklabels([0.05, 0.1, 0.2, 0.4, 0.8])
ax[1].scatter(results['rff']['x'][:-150], results['rff']['y'][:-150],
              marker='o', color=ccc['rff'], label='RFF', s=(100,))
ax[1].scatter(results['rff']['x'][-1], results['rff']['y'][-1],
              marker='X', color='orange', s=(100,))
ax[1].legend(fontsize=20)

fig.colorbar(cf, ax=ax[1])
if save_output:
    plt.savefig('./results/loss_landscapes_2.png')
    plt.savefig('./paper/figures/loss_landscapes_2.png')
    plt.savefig('./results/loss_landscapes_2.pdf')
    plt.savefig('./paper/figures/loss_landscapes_2.pdf')
plt.show()
