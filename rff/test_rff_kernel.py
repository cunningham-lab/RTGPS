import unittest
import torch
import numpy as np
from gpytorch.kernels.rr_rff import RR_RFF_Kernel
from gpytorch.likelihoods import GaussianLikelihood
from rff.rff_fns import compute_rff_z_gpt
from rff.rff_fns import OneOverJ
from experiments.experiment_fns import GPRegressionModel_RR_RFF


class TestKernelssrff(unittest.TestCase):

    def test_ss_estimator(self):
        tolerance = 1.e-5
        obs_num, data_dim = 6, 3
        train_ds = (torch.randn(obs_num, data_dim), torch.randn(obs_num, 1))
        probs = torch.tensor([1., 1. / 2., 1. / 3., 1. / 4.])
        probs /= probs.sum()

        likelihood, single_sample = GaussianLikelihood(), True
        dist_obj = OneOverJ(min_val=3, max_val=6 + 1)
        dist_obj.value_sampled = 4
        dist_obj.index_sampled = 1
        model = GPRegressionModel_RR_RFF(*train_ds, likelihood, dist_obj, single_sample)
        terms = torch.arange(4 + 3, 4, -1, dtype=torch.float32)
        approx = model.single_sample_estimator(terms)

        ans = torch.tensor(7.) + (torch.tensor(5.) - torch.tensor(6.)) / probs[1]

        diff = np.abs(ans.numpy() - approx.numpy())
        print('\nTEST: Single Sample computation')
        print(f'Diff       {diff:+1.3e}')
        self.assertTrue(expr=diff < tolerance)

        dist_obj.value_sampled = 3
        dist_obj.index_sampled = 0
        model = GPRegressionModel_RR_RFF(*train_ds, likelihood, dist_obj, single_sample)
        terms = torch.arange(4 + 3, 4, -1, dtype=torch.float32)
        approx = model.single_sample_estimator(terms)

        ans = torch.tensor(7.) + (torch.tensor(6.) - torch.tensor(7.)) / probs[0]

        diff = np.abs(ans.numpy() - approx.numpy())
        print(f'Diff       {diff:+1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_expand_z(self):
        tolerance = 1.e-5
        min_val = 2

        z1 = torch.tensor([[0, 1, 0, 0, 4, 5, 0, 0],
                           [8, 9, 0, 0, 12, 13, 0, 0]],
                          dtype=torch.float32).unsqueeze(0)
        z1 /= torch.sqrt(torch.tensor(2.))
        z2 = torch.tensor([[16, 17, 18, 0, 20, 21, 22, 0],
                           [0, 1, 2, 0, 4, 5, 6, 0]],
                          dtype=torch.float32).unsqueeze(0)
        z2 /= torch.sqrt(torch.tensor(3.))
        z3 = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15],
                           [16, 17, 18, 19, 20, 21, 22, 23]],
                          dtype=torch.float32).unsqueeze(0)
        z3 /= torch.sqrt(torch.tensor(4.))
        z_ans = torch.cat([z1, z2, z3], dim=0)
        sampled_z_num, obs_num = z_ans.shape[0], z_ans.shape[1]
        num_features = z_ans.shape[2] // 2

        ker = RR_RFF_Kernel(single_sample=True, min_val=torch.tensor(min_val))
        z1 = torch.arange(0, sampled_z_num * obs_num * num_features)
        z2 = torch.arange(0, sampled_z_num * obs_num * num_features)
        z = torch.cat([z1, z2], dim=-1)
        z = torch.reshape(z, shape=(sampled_z_num, obs_num, num_features * 2)).float()
        z_new = ker.expand_z(z)

        diff = np.linalg.norm(z_new.numpy() - z_ans.numpy())
        print('\nTEST: z expansion method')
        print(f'Diff       {diff:+1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_featurize(self):
        tolerance = 1.e-5
        diff = 0
        obs_num, data_dim = 2, 4
        num_features = 4
        sampled_z_num = 3

        x = torch.randn(obs_num, data_dim)
        ker = RR_RFF_Kernel(single_sample=True, min_val=torch.tensor(9.))
        num_dims = x.size(-1)
        ker._init_weights(num_dims, num_features)
        weights = ker.randn_weights
        z1 = ker._featurize(x, normalize=False)
        z = np.zeros(shape=(sampled_z_num, obs_num, num_features * 2), dtype=np.float32)
        for i in range(sampled_z_num):
            x_norm = (x / ker.lengthscale).detach().numpy()
            z[i, :, :] = compute_rff_z_gpt(x_norm, weights[i, :, :], num_features)
            z[i, :, :] *= np.sqrt(num_features)
        size = torch.Size([sampled_z_num, obs_num, num_features * 2])
        diff = np.linalg.norm(z1.detach().numpy() - z)

        print('\nTEST: z features creation and size')
        print(f'Diff       {diff:+1.3e}')
        self.assertTrue(expr=z1.shape == size)
        self.assertTrue(expr=diff < tolerance)

    def test_random_weights(self):
        obs_num, data_dim = 10, 4
        feature_num = 6
        sampled_z_num = 3

        x = torch.randn(obs_num, data_dim)
        ker = RR_RFF_Kernel(single_sample=True, min_val=torch.tensor(9.))
        num_dims = x.size(-1)
        ker._init_weights(num_dims, feature_num)
        size = torch.Size([sampled_z_num, data_dim, feature_num])

        print('\nTEST: Random weights shape')
        self.assertTrue(expr=ker.randn_weights.shape == size)


if __name__ == '__main__':
    unittest.main()
