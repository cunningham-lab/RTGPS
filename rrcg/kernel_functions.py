import torch
import numpy as np


def is_vector(vec):
    return vec.ndimension() == 1


def sqexp(x, y, sig2, ell):
    if is_vector(x):
        x = x.unsqueeze(-1)
    if is_vector(y):
        y - y.unsqueeze(-1)
    sqdist = torch.sum(((x.unsqueeze(-2) - y.unsqueeze(-3))/ell)**2, dim=-1)
    k = sig2*torch.exp(-sqdist / 2)
    return k


def matern(x, y, sig2, ell, nu=0.5):
    if is_vector(x):
        x = x.unsqueeze(-1)
    if is_vector(y):
        y = y.unsqueeze(-1)
    sqdist = torch.sum((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2, dim=-1)  # / (ell*ell)
    if nu == .5:
        kmat = torch.exp(-torch.sqrt(sqdist) / ell)
    elif nu == 1.5:
        dp = np.sqrt(3) * torch.sqrt(sqdist) / ell
        kmat = (1 + dp) * torch.exp(-dp)
    elif nu == 2.5:
        dp = np.sqrt(5) * torch.sqrt(sqdist) / ell
        kmat = (1 + dp + (5. / 3.) * sqdist / (ell ** 2)) * torch.exp(-dp)
    return sig2 * kmat
