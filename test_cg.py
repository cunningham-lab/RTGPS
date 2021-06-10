import torch
import gpytorch
import math
from kernels import *

from gpytorch.utils.linear_cg import linear_cg
torch.manual_seed(42)
sig2 = 1.0
lengthscale = 0.05
bsz = 200
train_x = torch.linspace(-3, 3, bsz)
train_y = torch.sin(train_x * (5 * math.pi)) * train_x + torch.randn_like(train_x) * 0.1
train_y = train_y.unsqueeze(1)

Kxx = matern(train_x, train_x, sig2, lengthscale, nu=1.5)


def linear_cg_rr(kxx, rhs, use_rr=False, dist_of_iter=None, max_iter=None, J=None,
                 max_tridiag_iter=None, tolerance=1e-5, normalize_rhs=True, preconditioner=None):
    if use_rr:
        assert dist_of_iter is not None
        if J is None:
            J = dist_of_iter.sample()
        else:
            assert max_iter <= dist_of_iter.N, "given J = {}, but maximum J allowed is {}".format(J, dist_of_iter.N)
        max_iter = J

    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)
    if normalize_rhs:
        rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
        rhs = rhs.div(rhs_norm)

    if preconditioner is None:
        preconditioner = lambda x: x

    initial_guess = torch.zeros_like(rhs)
    residual = rhs - kxx @ initial_guess
    batch_shape = residual.shape[:-2]

    # defining storage matrixes
    mul_storage = torch.empty_like(residual)
    alpha = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=residual.dtype, device=residual.device)
    beta = torch.empty_like(alpha)

    search_direction = residual.clone()
    solution = initial_guess.expand_as(residual).contiguous()

    if use_rr:
        rr_solution = solution.clone()
        rr_residual = residual.clone()

    precond_residual = preconditioner(residual)
    residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

    for k in range(max_iter):
        mvm = kxx @ search_direction
        torch.mul(search_direction, mvm, out=mul_storage)
        torch.sum(mul_storage, -2, keepdim=True, out=alpha)

        torch.div(residual_inner_prod, alpha, out=alpha)

        # update solution
        torch.addcmul(solution, alpha, search_direction, out=solution)

        # update residual
        torch.addcmul(residual, -alpha, mvm, out=residual)
        # update precond_redisual
        precond_residual = preconditioner(residual)


        if use_rr:
            # update solution
            prob_k = (1 - dist_of_iter.cdf(k))
            torch.div(alpha, prob_k, out=alpha)
            torch.addcmul(rr_solution, alpha, search_direction, out=rr_solution)
            #rr_solution = rr_solution + alpha * search_direction / prob_k

            # update residual
            torch.addcmul(rr_residual, -alpha, mvm, out=rr_residual)

            if rr_residual.norm() < tolerance:
                print("Terminating after {} iterations.".format(k+1))

        if residual.norm() < tolerance:
            print("Terminating after {} iterations.".format(k+1))
            break

        beta.resize_as_(residual_inner_prod).copy_(residual_inner_prod)

        # update residual_inner_product
        torch.mul(residual, precond_residual, out=mul_storage)
        torch.sum(mul_storage, -2, keepdim=True, out=residual_inner_prod)

        # update beta
        torch.div(residual_inner_prod, beta, out=beta)

        # update search direction
        search_direction.mul_(beta).add_(residual)
    if use_rr:
        print("Residual after {} iterations: {}".format(k + 1, rr_residual.norm()))
        if normalize_rhs:
            return rr_solution * rhs_norm
        return rr_solution

    print("Residual after {} iterations: {}".format(k+1, residual.norm()))
    if normalize_rhs:
        return solution * rhs_norm
    return solution

# test cg
print("solve using full cg")
num_cg = bsz
jitter = 1e-4
solve_linear_cg = linear_cg_rr(Kxx+jitter * torch.eye(bsz), train_y, max_iter=num_cg, normalize_rhs=False, tolerance=1e-25)

# exact solve
print("\nsolve using exact solve")
solve_exact, _ = torch.solve(train_y, Kxx + jitter * torch.eye(bsz))  # (M, 1)
residual = torch.matmul(Kxx, solve_exact) - train_y
print("residaul = {}".format(residual.norm()))


# test rr estimate

from dist_of_iterations import ExpDecayDist, UniformDist
N = 185
dist = ExpDecayDist(temp=0.01, N=N)
#dist = UniformDist(N=N)

num_samples = 1000
solve_rr_list = torch.empty(num_samples, *train_y.shape)
for i in range(num_samples):
    solve_linear_cg_rr = linear_cg_rr(Kxx+jitter * torch.eye(bsz), train_y, use_rr=True, dist_of_iter=dist, normalize_rhs=False, tolerance=1e-25)
    solve_rr_list[i] = solve_linear_cg_rr
residual_average = torch.matmul(Kxx, solve_rr_list.mean(dim=0))- train_y
print("\nResidual average = {}".format(residual_average.norm()))

diff = solve_linear_cg - solve_rr_list.mean(dim=0)
print(diff.norm())


