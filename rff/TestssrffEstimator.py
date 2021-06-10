import unittest
import math
import torch
import gpytorch as gpt
from matplotlib import pyplot as plt
from experiments.experiment_fns import GPRegressionModel
from experiments.experiment_fns import GPRegressionModel_RR_RFF
from experiments.experiment_fns import set_hypers
from experiments.truncation_fns import compute_invquad_logdet_rff
from rff.rff_fns import compute_gaussian_kernel
from rff.rff_fns import OneOverJ


class TestssrffEstimator(unittest.TestCase):

    def test_distributions(self):
        print('\nTEST: Truncation Distributions')
        tolerance = 1.e-1
        coeff = 2
        max_val = 5 * int(1.e1)
        min_val = 10
        step = 10
        total = (max_val - min_val) // step
        dist = OneOverJ(min_val=min_val, max_val=max_val, step=step, coeff=coeff)
        probs = [1. / ((step * i) ** coeff) for i in range(1, total + 1)]
        probs = torch.tensor(probs)
        probs /= torch.sum(probs)
        print('Check shapes')
        print(f'Shape: {dist.probs.shape[0]:2d}')
        self.assertTrue(expr=dist.probs.shape[0] == total)
        self.assertTrue(expr=probs.shape[0] == total)
        diff = torch.linalg.norm(probs - dist.probs)
        print(f'Diff:  {diff:+1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_ssrff_estimator_grads(self):
        # tolerance, sample_size = 1.e-2, int(1.e4)
        tolerance, sample_size = 1.e+1, int(1.e2)
        data_size, data_dim = 100, 5
        x, y = get_dataset1(data_size, data_dim)
        cases = [{'noise': 1.0, 'ls': 1.0, 'os': 1.0},
                 {'noise': 0.160930, 'ls': 0.705252, 'os': 1.0}]
        case = cases[1]
        # out = [torch.log(1. + torch.exp(i)) for i in model.parameters()]
        noise_grad, ls_grad, os_grad = compute_cholesky_grads(x, y, case)

        dist = OneOverJ(min_val=10, max_val=50)
        ng, lg = torch.zeros(sample_size), torch.zeros(sample_size)
        og = torch.zeros(sample_size)
        likelihood = gpt.likelihoods.GaussianLikelihood()
        model = GPRegressionModel_RR_RFF(x, y, likelihood, dist, single_sample=True)
        set_hypers(model, case['noise'], case['ls'], case['os'])
        mll = gpt.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for s in range(sample_size):
            optimizer.zero_grad()
            loss = model.RT_estimator(-mll(model(x), y))
            loss.backward()
            ng[s], lg[s], og[s] = [i.grad for i in model.parameters()]
        print('\nTEST: Unbiasedness in gradients')
        print_stats(noise_grad.numpy()[0], ng, text='noise grad ')
        print_stats(ls_grad.numpy(), lg, text='ls grad ')
        print_stats(os_grad.numpy()[0, 0], og, text='og grad ')
        tests = [(noise_grad, ng), (ls_grad, lg), (os_grad, og)]
        for test in tests:
            diff = torch.linalg.norm(test[0] - torch.mean(test[1]))
            self.assertTrue(expr=diff < tolerance)

    def test_loss_equivalence_between_methods(self):
        # TODO: improve this test, right now since the randn weights are not shared its
        # hard to know if the methods compute the same things
        tolerance = 1.e+1
        data_size, data_dim = 100, 5
        x, y = get_dataset1(data_size, data_dim)
        cases = [{'noise': 1.0, 'ls': 1.0, 'os': 1.0},
                 {'noise': 0.160930, 'ls': 0.705252, 'os': 1.0}]
        case = cases[1]
        dist = OneOverJ(min_val=10, max_val=11)
        likelihood = gpt.likelihoods.GaussianLikelihood()
        model = GPRegressionModel_RR_RFF(x, y, likelihood, dist, single_sample=True)
        set_hypers(model, case['noise'], case['ls'], case['os'])
        mll = gpt.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        loss = model.RT_estimator(mll(model(x), y))
        out = model.likelihood(model(x))
        logdet = out.lazy_covariance_matrix.logdet()
        invquad = out.lazy_covariance_matrix.inv_quad(y)
        invquad_debiased = model.RT_estimator(invquad).cpu()
        logdet_debiased = model.RT_estimator(logdet).cpu()
        loss_approx = compute_loss_from_invquad_logdet(invquad_debiased,
                                                       logdet_debiased, y)
        print('\nTEST: Loss equivalence computation')
        diff = torch.linalg.norm(loss - loss_approx)
        print(f'Loss 1 {loss_approx.detach().numpy():+1.3f}')
        print(f'Loss 2 {loss.detach().numpy():+1.3f}')
        print(f'Diff   {diff:+1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_ssrff_estimators(self):
        # tolerance, sample_size = 1.e+1, int(1.e4)
        tolerance, sample_size = 1.e+2, int(1.e2)
        min_val, max_val = 10, 50
        # coeff, step = 0.25, 1
        coeff, step = 1.5, 10
        data_size, data_dim = 100, 5
        x, y = get_dataset1(data_size, data_dim)
        cases = [{'noise': 1.0, 'ls': 1.0, 'os': 1.0},
                 {'noise': 0.160930, 'ls': 0.705252, 'os': 1.0}]
        case = cases[0]
        invquad_ans, logdet_ans = compute_invquad_logdet_cholesky(x, y, case)
        dist = OneOverJ(min_val=min_val, max_val=max_val, coeff=coeff, step=step)
        invquad_min, logdet_min = compute_invquad_logdet_rff(x, y, min_val, case)
        invquad_max, logdet_max = compute_invquad_logdet_rff(x, y, max_val, case)
        invquad, logdet = torch.zeros(sample_size), torch.zeros(sample_size)
        for s in range(sample_size):
            out = compute_invquad_logdet_ssrff(x, y, dist, case)
            invquad[s], logdet[s] = out
        diff_invquad = torch.linalg.norm(invquad_ans - torch.mean(invquad))
        diff_logdet = torch.linalg.norm(logdet_ans - torch.mean(logdet))
        print(f'\nTEST: Unbiasedness with coeff = {coeff:4.3f}, step = {step:2d}')
        print(f'invquad min {invquad_min:4.3f}')
        print(f'invquad max {invquad_max:4.3f}')
        print_stats(invquad_ans.numpy(), invquad, text='invquad')
        print(f'logdet min {logdet_min:4.3f}')
        print(f'logdet max {logdet_max:4.3f}')
        print_stats(logdet_ans.numpy(), logdet, text='logdet')
        self.assertTrue(expr=diff_invquad < tolerance)
        self.assertTrue(expr=diff_logdet < tolerance)

    def test_covariance_computation(self):
        tolerance = 1.e-5
        data_size, data_dim = 100, 5
        x, y = get_dataset1(data_size, data_dim)
        likelihood = gpt.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(x, y, likelihood)
        cases = [{'noise': 1.0, 'ls': 1.0, 'os': 1.0},
                 {'noise': 0.160930, 'ls': 0.705252, 'os': 1.0}]
        case = cases[0]
        print('\nTEST: Covariance Computation')
        for case in cases:
            set_hypers(model, case['noise'], case['ls'], case['os'])
            ker = compute_gaussian_kernel(x.numpy(), sigma_2=case['ls'] ** 2)
            ker = torch.tensor(ker, dtype=torch.float32)
            ker += case['noise'] * torch.eye(data_size)
            yhat = model(x)
            mean, covar = yhat.mean, yhat.lazy_covariance_matrix
            diff = y - mean
            covar += model.likelihood.noise_covar(shape=mean.shape)
            covar = covar.evaluate_kernel()
            covar = covar.evaluate()

            diff = torch.linalg.norm(ker - covar)
            print(f'Diff {diff:+1.3e}')
            self.assertTrue(expr=diff < tolerance)

    def test_explicit_implicit_loss_computation(self):
        tolerance = 1.e-5
        data_size, data_dim = 100, 5
        x, y = get_dataset1(data_size, data_dim)
        likelihood = gpt.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(x, y, likelihood)
        mll = gpt.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        cases = [{'noise': 1.0, 'ls': 1.0, 'os': 1.0},
                 {'noise': 0.160930, 'ls': 0.705252, 'os': 0.62923026}]

        print('\nTEST: Implicit Explicit Loss Computation')
        for case in cases:
            set_hypers(model, case['noise'], case['ls'], case['os'])
            yhat = model(x)
            loss = mll(yhat, y)

            invquad, logdet = compute_invquad_logdet(model, x, y, yhat)
            res = compute_loss_from_invquad_logdet(invquad, logdet, y)

            diff = torch.linalg.norm(res - loss)
            print(f'Diff {diff:+1.3e}')
            self.assertTrue(expr=diff < tolerance)


def plot_histogram(x, trueval, title):
    plt.figure()
    plt.title(title)
    plt.hist(x, alpha=0.5)
    plt.axvline(x=trueval, color='red', label='true')
    plt.axvline(x=torch.mean(x), color='black', label='approx')
    plt.legend()
    plt.show()


def compute_cholesky_grads(x, y, case):
    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(x, y, likelihood)
    set_hypers(model, case['noise'], case['ls'], case['os'])
    mll = gpt.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    loss = -mll(model(x), y)
    loss.backward()
    noise_grad, ls_grad, os_grad = [i.grad for i in model.parameters()]
    return noise_grad, ls_grad, os_grad


def compute_loss_from_invquad_logdet(invquad, logdet, diff):
    data_size = diff.shape[0]
    res = -0.5 * sum([invquad, logdet, diff.size(-1) * math.log(2 * math.pi)])
    res = res.div(data_size)
    return res


def compute_invquad_logdet_ssrff(x, y, dist, hypers):
    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = GPRegressionModel_RR_RFF(x, y, likelihood, dist, single_sample=True)
    set_hypers(model, hypers['noise'], hypers['ls'], hypers['os'])
    with torch.no_grad(), gpt.settings.debug(False):
        out = model.likelihood(model(x))
        logdet = out.lazy_covariance_matrix.logdet()
        invquad = out.lazy_covariance_matrix.inv_quad(y)
        invquad_debiased = model.RT_estimator(invquad).cpu()
        logdet_debiased = model.RT_estimator(logdet).cpu()
    return invquad_debiased, logdet_debiased


def compute_invquad_logdet_cholesky(x, y, hypers):
    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(x, y, likelihood)
    set_hypers(model, hypers['noise'], hypers['ls'], hypers['os'])
    invquad, logdet = compute_invquad_logdet_from_model(x, y, model)
    return invquad, logdet


def compute_invquad_logdet_from_model(x, y, model):
    with torch.no_grad():
        yhat = model(x)
        invquad, logdet = compute_invquad_logdet(model, x, y, yhat)
    return invquad, logdet


def compute_invquad_logdet(model, x, y, yhat):
    mean, covar = yhat.mean, yhat.lazy_covariance_matrix
    diff = y - mean
    covar += model.likelihood.noise_covar(shape=mean.shape)
    covar = covar.evaluate_kernel()
    invquad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)
    return invquad, logdet


def print_stats(value, estimand, text):
    diff = torch.linalg.norm(torch.tensor(value) - torch.mean(estimand))
    text = text.ljust(20, ' ')
    print('+' * 50)
    print(text + f'{value:+1.3f}')
    print(text[:-11] + f'ss (mean)  {torch.mean(estimand).numpy():+1.3f}')
    print(text[:-11] + f'ss (std)   {torch.var(estimand).sqrt().numpy():+1.3f}')
    print(text[:-11] + f'ss (min)   {torch.min(estimand).numpy():+1.3f}')
    print(text[:-11] + f'ss (max)   {torch.max(estimand).numpy():+1.3f}')
    print('Diff ' + text[:-5] + f'{diff:+1.3e}')


def get_dataset1(data_size, data_dim):
    x = torch.rand(size=(data_size, data_dim), dtype=torch.float32)
    y = torch.sqrt(x) + 0.01 * torch.randn(size=(data_size, data_dim))
    y = torch.sum(y, dim=1)
    return (x, y)


if __name__ == '__main__':
    unittest.main()
