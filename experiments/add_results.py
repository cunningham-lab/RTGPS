from experiments.utils import load_results
from experiments.utils import save_results

file_path = './results/all_results_exact_bike.pkl'
results = load_results(file_path)

files = {
    'cg1': './results/evolution/loss_pol_cg_20.pkl',
    'cg2': './results/evolution/loss_pol_cg_40.pkl',
    'rrcg': './results/evolution/loss_pol_rrcg_20.pkl',
    'rrcg2': './results/evolution/loss_pol_rrcg_40.pkl',
    'chol_cg': './results/evolution/loss_pol_rrcg_40.pkl',
}

# x = load_results('./results/saved/ssrff2_exact_loss.pkl')
# results['ssrff2'] = {}
# results['ssrff2']['exact_loss'] = x
# results['ssrff2']['loss'] = x
# y = load_results('./results/cholesky/loss_bike_2021_02_04_2339_73284.pkl')
# results['chol_cg'] = {}
# results['chol_cg']['exact_loss'] = y['loss']
# results['chol_cg']['loss'] = y['loss']

for k, f in files.items():
    x = load_results(f)
    results[k] = {}
    results[k]['loss'] = x['opt_loss']
    results[k]['exact_loss'] = x['exact_loss']
save_results(results, file_path)
