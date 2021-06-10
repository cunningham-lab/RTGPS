import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from experiments.utils import get_variable_np_array_from_log_file

inputs = {
    # 'cholesky': './results/cholesky/pol/loss_pol_2021_01_22_1930_30632.log',
    'ssrff': '/home/pure/Downloads/crash/loss_pol_2021_01_26_2232_56030.log',
    'rff1': '/home/pure/Downloads/crash/loss_pol_rff200.log',
    'rff2': '/home/pure/Downloads/crash/loss_pol_rff700.log',
}
results = {}
max_iter = []
for key, input_file in inputs.items():
    ls = get_variable_np_array_from_log_file(variable_name='ls',
                                             path_to_file=input_file)
    noise = get_variable_np_array_from_log_file(variable_name='noise',
                                                path_to_file=input_file)
    os = get_variable_np_array_from_log_file(variable_name='os',
                                             path_to_file=input_file)
    loss = get_variable_np_array_from_log_file(variable_name='loss',
                                               path_to_file=input_file)
    max_iter.append(len(loss))
    results.update({key: {'ls': ls, 'os': os, 'noise': noise, 'loss': loss}})
max_iter = np.max(max_iter)


# #a6cee3 #1f78b4 #b2df8a #33a02c #fb9a99
colors = {'ssrff': '#a6cee3', 'rff1': '#b2df8a', 'rff2': '#1f78b4'}
# colors = {'rff1': '#b2df8a', 'rff2': '#1f78b4'}
# colors = {'ssrff': '#a6cee3', 'rff': '#b2df8a', 'cholesky': '#1f78b4'}
# colors = {'rff': '#b2df8a', 'cholesky': '#1f78b4'}
# colors = {'cholesky': '#1f78b4'}
variables = ['loss', 'ls', 'os', 'noise']
# variables = ['os']

# plt.figure()
# var = variables[0]
# plt.title(var)
# key = 'ssrff'
# df = pd.DataFrame(results[key][var])
# rolling = np.array(df.rolling(2000).mean())
# # plt.plot(results[key][var], color=colors[key], label=key)
# plt.plot(results['rff'][var], color=colors['rff'], label='rff')
# plt.plot(rolling, color='#fb9a99', label='ssrff rolling')
# plt.axhline(results['rff'][var][-1], color='black')
# plt.ylim([-1, 1])
# plt.legend()
# plt.show()
# plt.figure()

# var = variables[3]
# plt.title(var)
# key = 'ssrff'
# plt.plot(results[key][var], color=colors[key], label=key)
# plt.plot(results['rff'][var], color=colors['rff'], label='rff')
# plt.axhline(results['rff'][var][-1], color='black')
# plt.legend()
# plt.show()

for var in variables:
    plt.figure()
    plt.title(var)
    for key, color in colors.items():
        res = results[key][var]
        if len(res) < max_iter:
            diff = max_iter - len(res)
            a1 = res
            a2 = res[-1] * np.ones(diff)
            res = np.concatenate((a1, a2))
        plt.plot(res, color=color, label=key)
    plt.legend()
    plt.savefig('./results/evolution/' + var + '.png')
    plt.show()
