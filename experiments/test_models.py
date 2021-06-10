import torch
import gc
import time
from experiments.load_data import get_train_test_data
from experiments.experiment_fns import get_test_rmse_nll_gp
from experiments.utils import start_all_logging_instruments
from experiments.utils import get_available_logs
from experiments.utils import get_hypers
from experiments.utils import print_time_taken
from experiments.utils import load_results
from experiments.utils import get_variable_np_array_from_log_file

datasets = {1: 'pol', 2: 'ele', 3: 'bike', 4: 'kin', 5: 'protein',
            6: 'keggd', 7: 'slice', 8: 'keggu', 9: '3droad',
            10: 'song', 11: 'buzz', 12: 'house'}
# selected_datasets = [3, 2, 1]
selected_datasets = [6]
results_type = 'rff_ard_results'
read_pkl = True
# results_type = 'rff_results'
# read_pkl = False
run_sample = False
check_only = False
use_cuda = torch.cuda.is_available()
variables = ['ls', 'os', 'noise']
t0 = time.time()
for idx in selected_datasets:
    dataset_name = datasets[idx]
    train_ds, test_ds, valid_ds = get_train_test_data('./datasets/', dataset_name,
                                                      run_sample, use_cuda)
    file_path = './results/' + results_type + '/' + dataset_name + '/'
    logs = get_available_logs(file_path)
    settings = {'cuda': use_cuda, 'dataset_name': dataset_name,
                'train_obs': train_ds[0].shape[0], 'test_obs': test_ds[0].shape[0]}
    for log in logs:
        log_settings = settings.copy()
        hypers = get_hypers(file_path + log)
        log_settings.update(hypers)
        log_settings.update({'file_path': file_path + log})
        if read_pkl:
            res = load_results(file_path + log[:-4] + '.pkl')
            for var in variables:
                aux = res[var][:, -1] if var == 'ls' else res[var][-1]
                log_settings.update({var: aux})
        else:
            for var in variables:
                v = get_variable_np_array_from_log_file(var, file_path + log)
                log_settings.update({var: v[-1]})

        if not check_only:
            logger = start_all_logging_instruments(log_settings,
                                                   results_path='./logs/rmse_')
            rmse, nll = get_test_rmse_nll_gp(train_ds, test_ds, log_settings, logger)
            logger.info(f'Test RMSE: {rmse:4.5f} | Test NLL: {nll:4.5f}')
            rmse, nll = get_test_rmse_nll_gp(train_ds, valid_ds, log_settings, logger)
            logger.info(f'Valid RMSE: {rmse:4.5f} | Valid NLL: {nll:4.5f}')
            if use_cuda:
                # print(torch.cuda.memory_allocated())
                gc.collect()
                torch.cuda.empty_cache()
print_time_taken(t0, time.time())
