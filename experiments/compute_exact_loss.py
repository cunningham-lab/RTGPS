import time
from experiments.plot_fns import get_exact_loss_from_hypers_logged
from experiments.utils import save_results
from experiments.utils import print_time_taken
from experiments.utils import load_results

save_results_all = True
compute_exact_loss = True
run_sample = False
selected_models_for_exact_loss = ['ssrff2']
results = load_results(input_file='./results/all_results.pkl')

if compute_exact_loss:
    t0 = time.time()
    for key in selected_models_for_exact_loss:
        tic = time.time()
        ls = results[key]['ls'][0, :]
        noise, os = results[key]['noise'], results[key]['os']
        results[key]['exact_loss'] = get_exact_loss_from_hypers_logged(ls,
                                                                       noise,
                                                                       os,
                                                                       run_sample,
                                                                       'bike')

        if save_results_all:
            save_results(results[key]['exact_loss'],
                         output_file='./results/' + key + '_exact_loss.pkl')
        print_time_taken(tic, time.time(), text=f'Loop for {key.upper()} ' + 'took')
    print_time_taken(t0, time.time())
else:
    for k in selected_models_for_exact_loss:
        results[k]['exact_loss'] = results[k]['loss']

if save_results_all:
    save_results(results, output_file='./results/all_results_exact_bike.pkl')
