import time
from experiments.utils import get_vars_from_pkl
from experiments.utils import save_results
from experiments.utils import print_time_taken

inputs = {
    # 'cholesky': './results/evolution/loss_bike_cholesky.pkl',
    # 'rff1': './results/evolution/loss_bike_rff_1000.pkl',
    # 'rff2': './results/evolution/loss_bike_rff_1500.pkl',
    # 'ssrff': './results/evolution/loss_bike_ssrff_1000.pkl',
    'ssrff2': './results/evolution/loss_bike_ssrff_1500.pkl',
}
selected_models_for_exact_loss = ['rff1', 'rff2', 'ssrff', 'cholesky']
save_results_all = True
results = {}
t0 = time.time()
for model, input_file in inputs.items():
    logged_vars = get_vars_from_pkl(input_file)
    results.update({model: logged_vars})
print_time_taken(t0, time.time(), text='Reading logs took')

results['cholesky']['exact_loss'] = results['cholesky']['loss']
if save_results_all:
    save_results(results, output_file='./results/all_results.pkl')
