import torch
import gc
import time
from itertools import product
from experiments.load_data import get_train_data
from experiments.experiment_fns import fit_gp
from experiments.utils import start_all_logging_instruments
from experiments.utils import print_time_taken

run_sample = True
iters = int(1.e1)
seeds = [5328]
datasets = {1: 'pol', 2: 'ele', 3: 'bike', 4: 'kin', 5: 'protein'}
lrs = [0.01]
coeffs_steps = [(0.5, 10), (1.5, 100)]
use_cuda = torch.cuda.is_available()
t0 = time.time()
criteria = [c for c in product(lrs, coeffs_steps)]
experiments, i = {}, 1
for _, dataset_name in datasets.items():
    run_settings = {}
    for lr, trun in criteria:
        run_settings = {'model_name': 'ssrff', 'lr': lr, 'warmup': False,
                        'optimizer': 'Adam', 'dataset_name': dataset_name,
                        'total_iters': iters, 'truncation_name': 'onej',
                        'trunc_settings': {'min_val': 100, 'max_val': 1000,
                                           'coeff': trun[0], 'step': trun[1]}}
        experiments.update({i: run_settings})
        i += 1
for _, v in experiments.items():
    train_ds = get_train_data('./datasets/', v['dataset_name'],
                              run_sample, int(1.e1), use_cuda)
    settings = v.copy()
    settings.update({'lr_wd': 0.5, 'obs_num': train_ds[0].shape[0],
                     'cuda': use_cuda})
    settings.update({'seed': ''})
    settings.update({'mil': [int(0.850 * settings['total_iters']),
                             int(0.900 * settings['total_iters']),
                             int(0.950 * settings['total_iters'])]})
    for t in range(len(seeds)):
        settings['seed'] = seeds[t]
        logger = start_all_logging_instruments(settings, results_path='./logs/loss_')
        fit_gp(train_ds, settings, logger)

    if use_cuda:
        gc.collect()
        torch.cuda.empty_cache()

print_time_taken(t0, time.time())
