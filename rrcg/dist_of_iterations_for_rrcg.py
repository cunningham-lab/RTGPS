import torch
import numpy as np
from torch.distributions import Categorical


class RRDist(object):

    def __init__(self, N, probs=None, dtype=torch.float, device=torch.device('cpu')):
        """
        J takes value in 1, 2, ..., N
        Theoreti ally, probs[t] > 0, where t is the number of iterations after which cg converges
        """
        assert len(probs) == N, "N = {}, len(probs) = {}".format(N, len(probs))
        assert np.isclose(probs.sum().cpu().numpy(), 1), "probs must sum up to 1, but got probs.sum() = {}".format(
            probs.sum())

        self.N = N
        self.categorical = Categorical(probs=probs)
        self.probs = self.categorical.probs
        self.dtype = dtype
        self.device = device

    def sample(self, sample_size=()):
        return self.categorical.sample(sample_size) + 1

    def prob(self, J):
        assert J >= 1 and J <= self.N
        return self.categorical.probs[J - 1]

    def cdf(self, J):
        # currently used by Luhuan for J starting from zero
        # compute P(X<=J) = sum_{i=1}^J P(X_i)
        assert J >= 0 and J <= self.N
        return torch.sum(self.categorical.probs[:J])

    @property
    def mean(self):
        return torch.sum(self.categorical.probs * torch.arange(self.N, dtype=self.dtype, device=self.device)) + 1

    @property
    def var(self):
        mean_sq = self.mean ** 2
        Xsq = torch.arange(1, self.N + 1, dtype=self.dtype, device=self.device) ** 2
        mean_Xsq = torch.sum(self.categorical.probs * Xsq)
        return mean_Xsq - mean_sq

    @property
    def std(self):
        return torch.sqrt(self.var)


class ExpDecayDist(RRDist):

    def __init__(self, temp, min=1, max=None, dtype=torch.float, device=torch.device('cpu')):
        """J takes value in 1, 2, ..., N"""
        assert min >= 1, min
        assert max is not None and max >= min, f"min={min}, max={max}"
        assert temp > 0
        self.temp = temp
        self.min = min
        self.max = max

        logprobs = - temp * torch.arange(max - min + 1, dtype=dtype, device=device)
        zero_probs = torch.zeros(min - 1, dtype=dtype, device=device)
        probs = torch.exp(logprobs - torch.logsumexp(logprobs, dim=0))
        probs = torch.cat([zero_probs, probs])

        super(ExpDecayDist, self).__init__(N=max, probs=probs, dtype=dtype, device=device)


class UniformDist(RRDist):

    def __init__(self, N, dtype=torch.float, device=torch.device('cpu')):
        probs = 1.0 / N * torch.ones(N, dtype=dtype, device=device)

        super(UniformDist, self).__init__(N=N, probs=probs, dtype=dtype, device=device)


class OneOverJ(RRDist):

    def __init__(self, min=1, max=None, do_sqrt=False, dtype=torch.float, device=torch.device('cpu')):
        assert min >= 1, min
        assert max is not None and max >= min, max
        self.min = min
        self.max = max
        self.do_sqrt = do_sqrt

        J_range = torch.arange(1, max - min + 1 + 1, dtype=dtype, device=device)

        if self.do_sqrt:
            J_range = J_range.sqrt()

        logprobs = torch.log(torch.ones((max - min + 1), dtype=dtype, device=device)) - \
                   torch.log(J_range)
        zero_probs = torch.zeros(min - 1, dtype=dtype, device=device)
        probs = torch.exp(logprobs - torch.logsumexp(logprobs, dim=0))  # normalize
        probs = torch.cat([zero_probs, probs])

        super(OneOverJ, self).__init__(N=max, probs=probs, dtype=dtype, device=device)


class Geometric(RRDist):
    "P(X=k) \propto p^k, k=min, ..., max"

    def __init__(self, p, min=1, max=None, dtype=torch.float, device=torch.device('cpu')):
        assert min >= 1, min
        assert max is not None and max >= min, max
        assert 0 < p < 1, p
        self.p = p
        self.min = min
        self.max = max

        J_range = torch.arange(1, max - min + 1 + 1, dtype=dtype, device=device)
        logprobs = J_range * np.log(p)
        zero_probs = torch.zeros(min - 1, dtype=dtype, device=device)
        probs = torch.exp(logprobs - torch.logsumexp(logprobs, dim=0))  # normalize
        probs = torch.cat([zero_probs, probs])

        super(Geometric, self).__init__(N=max, probs=probs, dtype=dtype, device=device)


# polynomial

if __name__ == '__main__':
    # test RR estimator

    torch.manual_seed(42)


    def rr_estimate(xs, dist_of_iter, J=None, verbose=False):
        if J is None:
            J = dist_of_iter.sample()
        maxiter = J
        if verbose:
            print("J = ", int(J))

        rr_x = 0
        for n in range(maxiter):
            x = xs[n]
            prob_n = (1 - dist_of_iter.cdf(n))
            rr_x = rr_x + x / prob_n

            # print(prob_n.numpy())

        return rr_x


    """
    N = 2
    #xs = torch.randn(N) * 10
    xs = torch.tensor([1., 2])
    print("xs = ", xs)
    y = torch.sum(xs)
    #dist = ExpDecayDist(temp=0.1, N=N)
    dist = UniformDist(N=N)
    #samples = dist.sample((1000,))
    #plt.hist(samples, bins=10);

    num_estimates = 100
    rr_ys = torch.empty(num_estimates)
    for i in range(num_estimates):
        rr_ys[i] = rr_estimate(xs, dist)

    print("true value = {}".format(y))
    print("rr estimate mean = {}".format(rr_ys.mean()))
    print("rr estimate variance = {}".format(rr_ys.var(unbiased=False)))
    print("rr estimate variance = {}".format(rr_ys.var(unbiased=True)))

    """
    ###############
    # now test a 'converging series'
    N = 100
    xs = 2 ** (- torch.arange(1, N, dtype=torch.float, device=device))
    y = xs.sum()

    dist = ExpDecayDist(temp=0.1, min=3, max=N, dtype=torch.float, device=device)

    num_estimates = 1000
    rr_ys = torch.empty(num_estimates)
    for i in range(num_estimates):
        rr_ys[i] = rr_estimate(xs, dist, verbose=True)

    print("true value = {}".format(y))
    print("rr estimate mean = {}".format(rr_ys.mean()))
    print("rr estimate variance = {}".format(rr_ys.var(unbiased=False)))
    print("rr estimate variance = {}".format(rr_ys.var(unbiased=True)))
