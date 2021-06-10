import torch
import gc
import time
from experiments.load_data import get_train_data
from experiments.experiment_fns import fit_gp
from experiments.utils import start_all_logging_instruments
from experiments.utils import print_time_taken

run_sample = True
datasets = {1: 'pol', 2: 'elevators', 3: 'bike', 4: 'kin', 5: 'protein',
            6: 'keggd', 7: 'slice', 8: 'keggu', 9: '3droad', 11: 'buzz'}
seeds = [5328]
# seeds = [5328, 5945, 8965, 49, 9337]

t0 = time.time()
use_cuda = torch.cuda.is_available()
specific_lr = {'keggd': 0.001, 'keggu': 0.001, 'buzz': 0.005}
criteria = {'rff_samples': [1000]}
total_iters = 10
experiments, i = {}, 1
for _, dataset_name in datasets.items():
    run_settings = {}
    for key, options in criteria.items():
        lr = specific_lr[dataset_name] if dataset_name in specific_lr.keys() else 0.01
        for opt in options:
            run_settings = {'model_name': 'rff', 'lr': lr,
                            'optimizer': 'Adam',
                            key: opt, 'dataset_name': dataset_name,
                            'total_iters': total_iters}
            experiments.update({i: run_settings})
            i += 1
for _, v in experiments.items():
    dataset_name = v['dataset_name']
    train_ds = get_train_data('./datasets/', dataset_name,
                              run_sample, int(1.e1), use_cuda)
    settings = v.copy()
    settings.update({'lr_wd': 0.5, 'obs_num': train_ds[0].shape[0], 'cuda': use_cuda})
    settings.update({'seed': ''})
    settings.update({'mil': [int(0.500 * settings['total_iters']),
                             int(0.800 * settings['total_iters']),
                             int(0.900 * settings['total_iters'])]})
    for t in range(len(seeds)):
        settings['seed'] = seeds[t]
        logger = start_all_logging_instruments(settings, results_path='./logs/loss_')
        fit_gp(train_ds, settings, logger)

    if use_cuda:
        # print(torch.cuda.memory_allocated())
        gc.collect()
        torch.cuda.empty_cache()

print_time_taken(t0, time.time())
