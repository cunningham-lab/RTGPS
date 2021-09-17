import numpy as np
import numba
import torch
from torch.distributions import Categorical


class TruncationDist:

    def __init__(self, probs, min_val, max_val, step):
        self.min_val = torch.tensor(min_val, dtype=torch.int)
        self.max_val = torch.tensor(max_val, dtype=torch.int)
        self.categorical = Categorical(probs)
        self.probs = self.categorical.probs
        self.step = step
        self.index_sampled = torch.tensor(0, dtype=torch.int)
        self.value_sampled = torch.tensor(0, dtype=torch.int)

    def sample(self, sample_size=()):
        self.index_sampled = self.categorical.sample(sample_size)
        self.value_sampled = self.min_val + self.step * (self.index_sampled + 1)

    def prob(self, index):
        return self.categorical.probs[index]


class OneOverJ(TruncationDist):
    def __init__(self, min_val=1, max_val=None, step=1, coeff=1,
                 dtype=torch.float, device=torch.device("cpu")):
        total = (max_val - min_val) // step + 1
        truncation_range = torch.arange(1, total, dtype=dtype, device=device)
        log_probs = -coeff * torch.log(truncation_range)
        probs = torch.exp(log_probs - torch.logsumexp(log_probs, dim=0))
        super().__init__(probs=probs, min_val=min_val, max_val=max_val, step=step)


def get_diffs(v):
    diffs = np.zeros(v.shape)
    diffs[0] = v[0]
    for i in range(1, v.shape[0]):
        diffs[i] = v[i] - v[i - 1]
    return diffs


def get_diffs_mat(v, sf):
    diffs = np.zeros(v.shape)
    diffs[0, :, :] = v[0, :, :]
    for i in range(1, v.shape[0]):
        diffs[i, :, :] = v[i, :, :] - v[i - 1, :, :]
        diffs[i, :, :] /= sf[i - 1]
    return diffs


def get_all_logdet(x, min_feature, diff2max, step):
    total = diff2max // step + 1
    logdet_hat = np.zeros(total)
    ker_hat = compute_rff_ker_hat(x, min_feature)
    logdet_hat[0] = np.log(np.linalg.det(ker_hat))
    for j in range(1, total):
        num_features = min_feature + j * step
        ker_hat = compute_rff_ker_hat(x, num_features)
        logdet_hat[j] = np.log(np.linalg.det(ker_hat))
    return logdet_hat


def get_logdets(x, prob, ss, min_feature, step):
    sample_size = ss.shape[0]
    logdet_hat = np.zeros(sample_size)
    for s in range(sample_size):
        j = ss[s]
        if j == 0:
            z = sample_rff_features(x, min_feature)
            ker_hat = compute_ker_upto(z, min_feature)
            logdet_hat0 = np.log(np.linalg.det(ker_hat))
            logdet_hat[s] = logdet_hat0 / prob[j]
        else:
            a0, a1 = step * j + min_feature, step * (j + 1) + min_feature
            z = sample_rff_features(x, num_features=a1)
            logdet_hat[s] = compute_ss_logdet(z, prob[j], a0, a1)
    return logdet_hat


def compute_ss_logdet(z, prob, min_feature, max_feature):
    ker0 = compute_ker_upto(z, min_feature)
    ker1 = compute_ker_upto(z, max_feature)
    diff = (np.log(np.linalg.det(ker1)) - np.log(np.linalg.det(ker0))) / prob
    return diff


def compute_logdet_rff(z, min_features, max_features):
    log_det_diff = np.empty(shape=max_features - min_features)
    ker1 = compute_ker_upto(z, min_features)
    log_det_diff[0] = np.log(np.linalg.det(ker1))
    for i in range(1, max_features - min_features):
        ker0 = compute_ker_upto(z, i + min_features)
        ker1 = compute_ker_upto(z, i + min_features + 1)
        log_det_diff[i] = np.log(np.linalg.det(ker1)) - np.log(np.linalg.det(ker0))
    return log_det_diff


def compute_ker_upto(z, i):
    ker = np.matmul(z[:, 0:2 * i], z[:, 0:2 * i].T)
    return ker


def compute_rff_ker_hat_vec(x, num_features, sample_size):
    data_dim = x.shape[1]
    w = np.random.normal(size=(num_features, data_dim, sample_size))
    z = compute_rff_z_vec(x, w, num_features)
    ker_hat = multiply_zz(z)
    return ker_hat


# @numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
# @numba.jit(nopython=True, parallel=True, cache=True)
@numba.jit(nopython=True, cache=True)
def multiply_zz(z):
    num_obs = z.shape[0]
    sample_size = z.shape[2]
    ker_hat = np.zeros(shape=(num_obs, num_obs, sample_size))
    # for s in numba.prange(sample_size):
    for s in range(sample_size):
        ker_hat[:, :, s] = z[:, :, s] @ z.T[s, :, :]
    return ker_hat


def compute_rff_ker_hat(x, num_features):
    z = sample_rff_features(x, num_features)
    ker_hat = np.matmul(z, z.T)
    return ker_hat


def sample_rff_features(x, num_features):
    data_dim = x.shape[1]
    w = np.random.normal(size=(num_features, data_dim))
    z = compute_rff_z(x, w, num_features)
    return z


def compute_rff_z(x, w, num_features):
    data_size = x.shape[0]
    z = np.empty(shape=(data_size, 2 * num_features))
    w_x = np.tensordot(x, w, axes=([1], [1]))
    z[:, 0::2] = np.cos(w_x)
    z[:, 1::2] = np.sin(w_x)
    return np.sqrt(1 / num_features) * z


def compute_rff_z_gpt(x, w, num_features):
    data_size = x.shape[0]
    z = np.empty(shape=(data_size, 2 * num_features))
    w_x = np.matmul(x, w)
    z = np.concatenate((np.cos(w_x), np.sin(w_x)), axis=1)
    return np.sqrt(1 / num_features) * z


def compute_rff_z_vec(x, w, num_features):
    sample_size = w.shape[2]
    data_size = x.shape[0]
    z = np.empty(shape=(data_size, 2 * num_features, sample_size))
    w_x = np.tensordot(x, w, axes=([1], [1]))
    z[:, 0::2] = np.cos(w_x)
    z[:, 1::2] = np.sin(w_x)
    return np.sqrt(1 / num_features) * z


# @numba.jit(nopython=True)
def compute_gaussian_kernel(x, sigma_2):
    data_dim = x.shape[0]
    ker = np.zeros(shape=(data_dim, data_dim))
    for i in range(data_dim):
        for j in range(data_dim):
            delta = np.sum((x[i, :] - x[j, :]) ** 2)
            ker[i, j] = np.exp(- delta / (2. * sigma_2))
    return ker
