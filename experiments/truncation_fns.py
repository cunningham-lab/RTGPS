import torch
from torch.distributions import Categorical
import gpytorch as gpt
from experiments.experiment_fns import GPRegressionModelRFF
from experiments.experiment_fns import set_hypers


def compute_deltas_from_rate(const, coeff, total):
    delta = torch.tensor([const / (i ** coeff) for i in range(1, total + 1)])
    delta2 = torch.pow(delta, 2)
    return delta2


def compute_probs_from_rate(const, coeff, total):
    probs = torch.tensor([const / (i ** coeff) for i in range(1, total + 1)])
    probs /= torch.sum(probs)
    return probs


def compute_variance_ss(delta2, probs):
    variance = torch.sum(delta2 * ((1. - probs) / probs))
    return variance


def compute_invquad_logdet_rff(x, y, num_rff_samples, hypers):
    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = GPRegressionModelRFF(x, y, likelihood, num_rff_samples)
    set_hypers(model, hypers['noise'], hypers['ls'], hypers['os'])
    with torch.no_grad(), gpt.settings.debug(False):
        out = model.likelihood(model(x))
        logdet = out.lazy_covariance_matrix.logdet()
        invquad = out.lazy_covariance_matrix.inv_quad(y)
    return invquad, logdet


def variance_loss(theta, delta2):
    m = torch.nn.Softmax(dim=0)
    probs = m(theta)
    ratio = (1. - probs) / probs
    loss = torch.sum(delta2 * ratio)
    return loss


def get_mean_approx(estimand, sample_size, start):
    approx = torch.zeros(sample_size - start)
    for s in range(sample_size - start):
        approx[s] = torch.mean(estimand[0:s + start])
    return approx


def get_samples_from_probs(probs, sample_size):
    dist = Categorical(probs)
    dist_samples = dist.sample((sample_size,))
    return dist_samples


def get_estimands_from_probs(probs, delta, sample_size, offset):
    dist = Categorical(probs)
    dist_samples = dist.sample((sample_size,))
    estimand = delta[dist_samples] / dist.probs[dist_samples] + offset
    return estimand
