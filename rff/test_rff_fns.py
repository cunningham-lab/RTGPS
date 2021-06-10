import unittest
import numpy as np
import torch
from scipy.stats import poisson
from scipy.stats import nbinom
import gpytorch as gpt
from rff.rff_fns import compute_gaussian_kernel
from rff.rff_fns import compute_rff_z
from rff.rff_fns import compute_rff_z_gpt
from rff.rff_fns import compute_rff_ker_hat
from rff.rff_fns import compute_logdet_rff
from rff.rff_fns import get_logdets
from rff.rff_fns import compute_rff_ker_hat_vec
from rff.rff_fns import get_diffs
from rff.rff_fns import get_all_logdet
from experiments.experiment_fns import compute_logdet_invquad
from experiments.experiment_fns import GPRegressionModel
from experiments.experiment_fns import set_hypers


class TestRFFs(unittest.TestCase):

    def test_compute_rff_z_gpt(self):
        tolerance = 1.e-1
        data_dim = 2
        obs_num = 3
        x = np.arange(1, 7).reshape(obs_num, data_dim)
        weights = np.array([[1., 0., -1, 2.], [-1., 1., -2., -4.]])
        num_features = weights.shape[1]
        w_x = np.zeros(shape=(obs_num, num_features))
        for i in range(obs_num):
            for j in range(num_features):
                w_x[i, j] = np.dot(weights[:, j], x[i, :])

        z_ans = np.concatenate((np.cos(w_x), np.sin(w_x)), axis=1)
        z = compute_rff_z_gpt(x, weights, num_features)
        z *= np.sqrt(num_features)
        diff = np.linalg.norm(z_ans - z)
        row_check = np.linalg.norm(w_x[0, :] -
                                   np.array([-1., 2., - 5., -6.]))
        print('\nTEST: RFF z computation')
        print(f'Diff       {diff:+1.3e}')
        self.assertTrue(expr=row_check < tolerance)
        self.assertTrue(expr=diff < tolerance)

    def test_gpt_cholesky(self):
        tolerance = 1.e-1
        data_size, data_dim = 10, 5
        train_ds = get_dataset1(data_size, data_dim)
        ker = compute_gaussian_kernel(train_ds[0].numpy(), sigma_2=1.0)
        logdet = np.log(np.linalg.det(ker + 0.001 * np.eye(data_size)))
        alpha = np.linalg.solve(ker, train_ds[1])
        invquad = np.dot(train_ds[1], alpha)
        hypers = {'noise_scale': 0.001, 'ls': 1.0, 'output_scale': 1.0}

        likelihood = gpt.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(*train_ds, likelihood)
        set_hypers(model, **hypers)
        out = compute_logdet_invquad(model, train_ds)
        logdet_hat = out[0].item()
        invquad_hat = out[1].item()

        approx_logdet = np.mean(logdet_hat)
        diff_logdet = np.abs(logdet - approx_logdet)
        approx_invquad = np.mean(invquad_hat)
        diff_invquad = np.abs(invquad - approx_invquad)
        print('\nTEST: GPyTorch Cholesky logdet comparison')
        print(f'logdet            {logdet:+3.3f}')
        print(f'logdet_approx     {approx_logdet:+3.3f}')
        print(f'Diff              {diff_logdet:+1.3e}')
        print(f'invquad           {invquad:+3.3f}')
        print(f'invquad_approx    {approx_invquad:+3.3f}')
        print(f'Diff              {diff_invquad:+1.3e}')
        self.assertTrue(expr=diff_logdet < tolerance)
        self.assertTrue(expr=diff_invquad < tolerance)

    def test_ss_logdet_negbin(self):
        tolerance = 1.e+2
        data_size, data_dim = 10, 5
        x = np.random.uniform(size=(data_size, data_dim))
        ker = compute_gaussian_kernel(x, sigma_2=1.0)
        logdet = np.log(np.linalg.det(ker))
        # min_feature, step, max_feature = int(1.e2), 10, int(1.e3)
        # sucess_n, proba_fail = 1, 0.3
        # sample_size = int(1.e2)
        min_feature, step, max_feature = int(1.e3), 100, int(1.e4)
        sucess_n, proba_fail = 1, 0.5
        sample_size = int(1.e3)
        diff2max = max_feature - min_feature
        logdet_hat = get_all_logdet(x, min_feature, diff2max, step)
        deltas = get_diffs(logdet_hat)
        probs = nbinom.pmf(np.arange(deltas.shape[0]), n=sucess_n, p=proba_fail)
        deltas /= probs
        ss = nbinom.rvs(n=sucess_n, p=proba_fail, size=sample_size)
        x = deltas[ss]
        approx = np.mean(x)
        diff = np.abs(logdet - approx)
        print('\nTEST: RFF SS log det negbin')
        print(f'Diff       {diff:+1.3e}')
        print(f'Res        {logdet:1.3e}')
        print(f'Approx     {approx:1.3e}')
        print(f'MaxT       {np.max(ss):3d}')
        print(f'MaxTApprox {logdet_hat[np.max(ss)]:+1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_zero_slowly_negbin(self):
        tolerance = 1.e-2
        n, start, step = int(1.e5), int(1.e3), 100
        gamma = np.array([-1 / i for i in range(start, start + n, step)])
        deltas = get_diffs(gamma)
        sample_size = int(1.e3)
        sucess_n, proba_fail = 1, 0.1
        probs = nbinom.pmf(np.arange(deltas.shape[0]), n=sucess_n, p=proba_fail)
        ss = nbinom.rvs(n=sucess_n, p=proba_fail, size=sample_size)
        deltas /= probs
        x = deltas[ss]
        approx = np.mean(x)
        diff = np.abs(approx - 0.0)
        print('\nTEST: Below zero approx at rate 1/n - negbin')
        print(f'Approx     {approx:+1.3e}')
        print(f'Midpoint   {0.5 * (np.max(x) + np.min(x)):+1.3e}')
        print(f'Variance   {np.std(x):+1.3e}')
        print(f'Min        {np.min(x):+1.3e}')
        print(f'Max        {np.max(x):+1.3e}')
        print(f'MaxT       {np.max(ss):+1.3e}')
        print(f'MaxTApprox {-1 / np.max(ss):+1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_zero_est(self):
        tolerance = 1.e+0
        gamma = np.array([-3.0, -2.0, -1.0, 0.0])
        # p1 = 0.25
        p1 = 0.5  # best
        p = (1 - p1) / 3
        probs = np.array([p1, p, p, p])
        deltas = get_diffs(gamma)
        diff = np.linalg.norm(deltas - np.array([-3.0, 1.0, 1.0, 1.0]))
        self.assertTrue(expr=diff < 1.e-10)
        deltas /= probs
        sample_size = int(1.e2)
        j = np.random.choice(a=gamma.shape[0], size=sample_size, p=probs)
        x = deltas[j]
        approx = np.mean(x)
        diff = np.abs(approx - 0.0)
        print('\nTEST: Below zero approx')
        print('Using a balanced sampling reduces var and midpoint is spot on')
        print(f'Approx     {approx:+1.3e}')
        print(f'Midpoint   {0.5 * (np.max(x) + np.min(x)):+1.3e}')
        print(f'Variance   {np.std(x):+1.3e}')
        print(f'Min        {np.min(x):+1.3e}')
        print(f'Max        {np.max(x):+1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_ss_logdet_poisson(self):
        tolerance = 1.e+2
        data_size, data_dim = 10, 5
        x = np.random.uniform(size=(data_size, data_dim))
        ker = compute_gaussian_kernel(x, sigma_2=1.0)
        logdet = np.log(np.linalg.det(ker))
        min_feature = 1000
        step = 1
        poisson_mean = 1

        sample_size = int(1.e3)
        ss = poisson.rvs(mu=poisson_mean, size=sample_size)
        prob = poisson.pmf(ss, mu=poisson_mean)
        logdet_hat = get_logdets(x, prob, ss, min_feature, step)
        approx = np.mean(logdet_hat)
        diff = np.abs(logdet - approx)
        print('\nTEST: RFF SS log det poisson')
        print(f'Diff     {diff:+1.3e}')
        print(f'Res      {logdet:1.3e}')
        print(f'Approx   {approx:1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_rff_logdet_diffs(self):
        tolerance = 1.e-0
        N, data_dim = 10, 5
        num_features = int(1.e3)
        x = np.random.uniform(size=(N, data_dim))
        ker = compute_gaussian_kernel(x, sigma_2=1.0)
        logdet = np.log(np.linalg.det(ker))
        w = np.random.normal(size=(num_features, data_dim))
        z = compute_rff_z(x, w, num_features)
        logdet_diffs = compute_logdet_rff(z, num_features - 2, num_features)
        logdet_hat = np.sum(logdet_diffs)
        diff = np.abs(logdet - logdet_hat)
        print('\nTEST: RFF logdet diffs')
        print(f'\nDiff        {diff:+1.3e}')
        print(f'Res         {logdet:1.3e}')
        print(f'Approx      {logdet_hat:1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_rff_logdet_approx(self):
        tolerance = 1.e-1
        N, D = 10, 5
        num_features = int(1.e2)
        sample_size = int(1.e2)
        x = np.random.uniform(size=(N, D))
        ker = compute_gaussian_kernel(x, sigma_2=1.0)
        logdet = np.log(np.linalg.det(ker))
        # ker_hat = compute_rff_ker_hat(x, num_features)
        ker_hat = np.zeros(shape=(sample_size, N, N))
        for i in range(sample_size):
            ker_hat[i, :, :] = compute_rff_ker_hat(x, num_features)
        ker_hat = np.mean(ker_hat, axis=0)
        logdet_hat = np.log(np.linalg.det(ker_hat))
        diff = np.abs(logdet - logdet_hat)
        print('\nTEST: RFF logdet approx')
        print(f'\nDiff     {diff:+1.3e}')
        print(f'Res      {logdet:1.3e}')
        print(f'Approx   {logdet_hat:1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_rffs(self):
        tolerance = 1.e-3
        N, D = 10, 5
        sigma_2 = 10.0
        num_features = 100
        sample_size = int(1.e4)
        x = np.random.uniform(size=(N, D))
        ker = compute_gaussian_kernel(x, sigma_2)
        x = x / np.sqrt(sigma_2)
        ker_hat = compute_rff_ker_hat_vec(x, num_features, sample_size)
        ker_hat_mean = np.mean(ker_hat, axis=2)
        diff_mean = np.linalg.norm(ker - ker_hat_mean) / np.linalg.norm(ker)
        print('\nTEST: RFF Kernel Approx')
        print(f'\nMean Diff {diff_mean:1.3e}')
        self.assertTrue(expr=diff_mean < tolerance)


def get_dataset1(data_size, data_dim):
    x = np.random.uniform(size=(data_size, data_dim))
    y = np.sqrt(x[:, 0]) + 0.01 * np.random.normal(size=x.shape)[:, 0]
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return (x, y)


if __name__ == '__main__':
    unittest.main()
