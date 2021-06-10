import torch
import numpy as np
import fire
from experiments.load_data import get_train_data
from experiments.experiment_fns import fit_gp
from experiments.utils import start_all_logging_instruments


def run(dataset, model_name='rff_ard', seed=5348, total_n=-1, rff_features_n=700,
        total_iters=500, lr=0.01):

    torch.manual_seed(seed)
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()

    run_sample, sample_size = (False, 1) if total_n == -1 else (True, total_n)
    train_ds = get_train_data('./datasets/', dataset, run_sample, sample_size,
                              use_cuda)
    settings = {
        'model_name': model_name, 'warmup': False,
        'rff_samples': rff_features_n,
        'total_iters': total_iters, 'lr': lr, 'lr_wd': 0.5, 'optimizer': 'Adam',
        'obs_num': train_ds[0].shape[0], 'dataset_name': dataset,
        'cuda': use_cuda
    }
    settings.update({'seed': seed,
                     'truncation_name': 'onej',
                     'trunc_settings': {'min_val': rff_features_n, 'max_val': 1500,
                                        'coeff': 1., 'step': 100}})
    settings.update({'mil': [int(0.5 * settings['total_iters']),
                             int(0.8 * settings['total_iters']),
                             int(0.9 * settings['total_iters'])]})

    logger = start_all_logging_instruments(settings, results_path='./logs/loss_')
    fit_gp(train_ds, settings, logger)


if __name__ == "__main__":
    fire.Fire(run)
