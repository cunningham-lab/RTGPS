import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from experiments.utils import get_vars_from_logs

input_file = '/home/pure/Downloads/crash/loss_pol_2021_01_26_2232_56030.log'
# input_grid_file = './results/exact_loss_landscape_100.csv'
input_grid_file = './results/exact_loss_landscape_1000.csv'
output_graph = './results/loss_trajectory.png'

exact_df = pd.read_csv(input_grid_file).drop("outputscale", axis=1)
exact_df = exact_df.set_index(["lengthscale", "noise"]).drop_duplicates().reset_index()
ls = exact_df["lengthscale"]
num_ls = len(ls.unique())
noise = exact_df["noise"]
num_noise = len(noise.unique())

ls_grid = ls.values.reshape((num_ls, num_noise))
noise_grid = noise.values.reshape((num_ls, num_noise))
mll = exact_df["mll"].values.reshape((num_ls, num_noise))

results = get_vars_from_logs(input_file)
ls = results['ls']
noise = results['noise']

# points_ind = np.round(np.linspace(0, ls.shape[0] - 1, num=100)).astype(np.int).tolist()
total = ls.shape[0]
points_ind = np.round(np.linspace(0, total - 1, num=total)).astype(np.int).tolist()
num_points = len(points_ind)
x = np.zeros(num_points)
y = np.zeros(num_points)

for i, points in enumerate(points_ind):
    x[i] = ls[points]
    y[i] = noise[points]

fig, ax = plt.subplots(1, 1)
cf = ax.contourf(ls_grid, noise_grid, np.clip(mll, -1., None), level=50)
fig.colorbar(cf)
ax.set(xlabel="Lengthscale", ylabel="Noise", title="Exact GP Marginal Log Lik.")
ax.set_xlim((0.01, 1.))
ax.set_ylim((0.01, 1.))
plt.scatter(x, y, marker='o', color='orange')
plt.savefig(output_graph)
plt.show()
