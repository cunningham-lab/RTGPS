import numpy as np
import torch
import time
from datetime import datetime
import gpytorch
from rff.rff_fns import OneOverJ
from experiments.experiment_fns import GPRegressionModel
from experiments.experiment_fns import set_hypers
from experiments.experiment_fns import build_RFF_model, build_RR_RFF_model
from experiments.experiment_fns import run_rr_rounds, run_rff_rounds
from experiments.experiment_fns import compute_logdet_invquad
from experiments.load_data import load_uci_data_ap
from experiments.utils import print_time_taken
from experiments.utils import save_results

use_cuda = torch.cuda.is_available()
print(f'Using CUDA: {use_cuda}')
save_output = True
# num_rr_rounds = 1 * int(1.e1)
# min_val = torch.linspace(10, 40, 4, dtype=torch.int)
# train_n = int(1.e2)
num_rr_rounds = 1 * int(1.e4)
min_val = torch.linspace(100, 500, 10, dtype=torch.int)
train_n = int(1.e3)
train_x, train_y, *_ = load_uci_data_ap('./datasets', 'pol', use_cuda)
train_x, train_y = train_x[:train_n, :], train_y[:train_n]
train_ds = (train_x, train_y)
# max_val = len(train_x) // 2
max_val = 600

likelihood_chol = gpytorch.likelihoods.GaussianLikelihood()
likelihood_chol = likelihood_chol.cuda() if use_cuda else likelihood_chol
exact_model = GPRegressionModel(*train_ds, likelihood_chol)
exact_model = exact_model.cuda() if use_cuda else exact_model
hypers = {'noise_scale': 0.0163, 'ls': 0.366, 'output_scale': 0.375}
set_hypers(exact_model, **hypers)
print(hypers)
log_det_exact, inv_quad_exact = compute_logdet_invquad(exact_model, train_ds)

results = {}
results["cholesky"] = (log_det_exact.cpu().numpy(), inv_quad_exact.cpu().numpy())
res_list_rr, res_list_rff, num_rff_samples = [], [], np.zeros_like(min_val)
t0 = time.time()
print('Started looping')
with torch.no_grad():
    for i in range(len(min_val)):
        tic = time.time()
        J_dist = OneOverJ(min_val=min_val[i].item(), max_val=max_val,
                          step=10., coeff=1.)
        rr_model = build_RR_RFF_model(train_ds=train_ds, dist_obj=J_dist,
                                      single_sample=True, hypers=hypers,
                                      use_cuda=use_cuda)
        output = run_rr_rounds(num_rr_rounds, rr_model, exact_model, train_ds, use_cuda)
        res_list_rr.append(output)

        print('Saved... mid res')
        results["ssrff"] = res_list_rr
        results["rff"] = res_list_rff
        results["num_samples"] = num_rff_samples
        time_stamp = str(0)
        save_results(results, './results/unbiased_rff_' + time_stamp + '.pkl')

        num_rff_samples[i] = np.ceil(np.mean(res_list_rr[i]['J']))
        rff_model = build_RFF_model(hypers, int(num_rff_samples[i]), train_ds, use_cuda)
        output = run_rff_rounds(num_rr_rounds, rff_model, train_ds, use_cuda)
        res_list_rff.append(output)
        print_time_taken(tic, time.time(), 'Loop took')
print_time_taken(t0, time.time())

results["ssrff"] = res_list_rr
results["rff"] = res_list_rff
results["num_samples"] = num_rff_samples
time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M")
# time_stamp = str(0)
save_results(results, './results/unbiased_rff_' + time_stamp + '.pkl')
