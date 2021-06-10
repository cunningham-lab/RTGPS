import warnings
from gpytorch.utils.warnings import NumericalWarning
from gpytorch import settings
import math
from rrcg.kernel_functions import *


def _default_preconditioner(x):
    return x.clone()


def linear_cg_rr(matmul_closure, rhs, n_tridiag=0,
                 use_rr=False, dist_of_iter=None, max_iter=None, max_rrcg_iter_list=None,
                 max_tridiag_iter=None, tolerance=None, stop_updating_after=1e-10, initial_guess=None,
                 preconditioner=None,
                 eps=1e-10,
                 verbose=False):
    # some default arguments
    num_rows = rhs.size(-2)
    if use_rr:
        assert max_rrcg_iter_list is not None
        assert dist_of_iter is not None

        sorted_max_rrcg_iter_list, sorted_max_rrct_iter_idx = torch.sort(max_rrcg_iter_list)

        sorted_curr_Jidx = 0  # current index of max-rrcg-iter
        curr_Jvalue = sorted_max_rrcg_iter_list[0] # current value of max-rrcg-iter
        original_curr_Jidx = sorted_max_rrct_iter_idx[0]

        max_iter = sorted_max_rrcg_iter_list[-1]
        rr_nsamples = len(max_rrcg_iter_list)
        if verbose:
            print("Using rr cg!")
            print("max_iter = {}".format(max_iter))
            print("max_rrcg_iter_list = {}".format(max_rrcg_iter_list))

        if n_tridiag:
            if max_tridiag_iter is None:
                max_tridiag_iter = max_iter
        else:
            max_tridiag_iter = 0

        assert max_iter <= num_rows, \
            "rr max iter shoule be less than num_rows, but got max_iter = {}, num_rows = {}".format(max_iter, num_rows)
    else:
        if max_iter is None:
            max_iter = settings.max_cg_iterations.value()
        if n_tridiag:
            if max_tridiag_iter is None:
                max_tridiag_iter = settings.max_lanczos_quadrature_iterations.value()
        else:
            max_tridiag_iter = 0

    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)

    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)
    if tolerance is None:
        if settings._use_eval_tolerance.on():
            tolerance = settings.eval_cg_tolerance.value()
        else:
            tolerance = settings.cg_tolerance.value()
    if preconditioner is None:
        preconditioner = _default_preconditioner
        precond = False
    else:
        precond = True

    if max_tridiag_iter > max_iter:
        raise RuntimeError("Getting a tridiagonalization larger than the number of CG iterations run is not possible!")

    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    # get some constants
    n_tridiag_iter = min(max_tridiag_iter, num_rows)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    # Get the norm of the rhs - used for convergence checks
    # Here we're going to make almost-zero norms actually be 1 (so we don't get divide-by-zero issues)
    # But we'll store which norms were actually close to zero
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)

    # normalize rhs. We'll un-normalize later
    rhs = rhs.div(rhs_norm)

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(initial_guess)
    batch_shape = residual.shape[:-2]

    # solution <- x_0
    solution = initial_guess.expand_as(residual).contiguous()

    # Check for NaNs
    if not torch.equal(residual, residual):
        raise RuntimeError("NaNs encountered when trying to perform matrix-vector multiplication")

    residual_norm = residual.norm(2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    # set n_iter to be max_iter, for now
    n_iter = max_iter
    if has_converged.all() and not n_tridiag:
        n_iter = 0
    else:
        precond_residual = preconditioner(residual)
        search_direction = precond_residual
        residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

        # defining storage matrices
        mul_storage = torch.empty_like(residual)
        alpha = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=residual.dtype, device=residual.device)
        beta = torch.empty_like(alpha)
        is_zero = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=torch.bool, device=residual_norm.device)

        if use_rr:
            rr_solution = solution.clone().expand((rr_nsamples, ) + solution.shape).contiguous()

        # Define tridiagonal matrices, if applicable
        if n_tridiag:
            t_mat = torch.zeros(
                n_tridiag_iter, n_tridiag_iter, *batch_shape, n_tridiag, dtype=alpha.dtype, device=alpha.device)

            alpha_tridiag_is_zero = torch.empty(*batch_shape, n_tridiag, dtype=torch.bool, device=t_mat.device)
            alpha_reciprocal = torch.empty(*batch_shape, n_tridiag, dtype=t_mat.dtype, device=t_mat.device)
            prev_alpha_reciprocal = torch.empty_like(alpha_reciprocal)
            prev_beta = torch.empty_like(alpha_reciprocal)

        update_tridiag = True
        last_tridiag_iter = 0

    tolerance_reached = False
    cg_solution_converged = False

    for k in range(n_iter):
        mvm = matmul_closure(search_direction)

        torch.mul(search_direction, mvm, out=mul_storage)
        torch.sum(mul_storage, -2, keepdim=True, out=alpha)

        # safe division
        torch.lt(alpha, eps, out=is_zero)
        alpha.masked_fill_(is_zero, 1)
        torch.div(residual_inner_prod, alpha, out=alpha)
        alpha.masked_fill_(is_zero, 0)

        # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
        alpha.masked_fill_(has_converged, 0)

        # update residual
        torch.addcmul(residual, -alpha, mvm, out=residual)

        # update precond_redisual
        precond_residual = preconditioner(residual)

        # update solution
        torch.addcmul(solution, alpha, search_direction, out=solution)

        if use_rr:
            # original solution: x_k = x_0 + alpha1 d_1  + alpha_2 d_2 + .... alpha_k d_k
            # rr solution: \tilde{x}_k = x_0 + alpha1 d1 / (1-cdf(0)) + alpha2 d2 / (1-cdf(1)) .... + alphak dk / (1-cdf(k-1))

            # check which updating rr_estimate for which truncation number J
            while k + 1 > curr_Jvalue:
                # move to next rr-estimate where the truncation_number =  curr_Jvalue
                previous_original_curr_Jidx = sorted_max_rrct_iter_idx[sorted_curr_Jidx]

                sorted_curr_Jidx += 1
                curr_Jvalue = sorted_max_rrcg_iter_list[sorted_curr_Jidx]
                original_curr_Jidx = sorted_max_rrct_iter_idx[sorted_curr_Jidx]

                # we need to copy accumulated sum from from previous
                #rr_solution[sorted_curr_Jidx] = rr_solution[sorted_curr_Jidx - 1].clone()
                rr_solution[original_curr_Jidx] = rr_solution[previous_original_curr_Jidx].clone()

            # update solution: rr_solution = rr_solution + alpha * search_direction / prob_k
            prob_k = (1 - dist_of_iter.cdf(k))
            #torch.addcmul(rr_solution[sorted_curr_Jidx], alpha/prob_k, search_direction, out=rr_solution[sorted_curr_Jidx])
            torch.addcmul(rr_solution[original_curr_Jidx], alpha / prob_k, search_direction,
                          out=rr_solution[original_curr_Jidx])

        beta.resize_as_(residual_inner_prod).copy_(residual_inner_prod)

        # update residual_inner_product
        torch.mul(residual, precond_residual, out=mul_storage)
        torch.sum(mul_storage, -2, keepdim=True, out=residual_inner_prod)

        # update beta -- do a safe division here
        torch.lt(beta, eps, out=is_zero)
        beta.masked_fill_(is_zero, 1)
        torch.div(residual_inner_prod, beta, out=beta)
        beta.masked_fill_(is_zero, 0)

        # update search direction
        search_direction.mul_(beta).add_(precond_residual)

        torch.norm(residual, 2, dim=-2, keepdim=True, out=residual_norm)
        residual_norm.masked_fill_(rhs_is_zero, 0)
        torch.lt(residual_norm, stop_updating_after, out=has_converged)

        #if k >= 10 and bool(residual_norm.mean() < tolerance) and not (n_tridiag and k < n_tridiag_iter):
        if bool(residual_norm.mean() < tolerance):
            cg_solution_converged = True
            print("cg converges after {} iterations".format(k+1))
            if not (n_tridiag_iter and k < n_tridiag_iter and update_tridiag):  # TODO: check this
                tolerance_reached = True
                break

        if n_tridiag and k < n_tridiag_iter and update_tridiag:
            alpha_tridiag = alpha.squeeze_(-2).narrow(-1, 0, n_tridiag)
            beta_tridiag = beta.squeeze_(-2).narrow(-1, 0, n_tridiag)
            torch.eq(alpha_tridiag, 0, out=alpha_tridiag_is_zero)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 1)
            torch.reciprocal(alpha_tridiag, out=alpha_reciprocal)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 0)

            if k == 0:
                t_mat[k, k].copy_(alpha_reciprocal)
            else:
                torch.addcmul(alpha_reciprocal, prev_beta, prev_alpha_reciprocal, out=t_mat[k,k])
                torch.mul(prev_beta.sqrt_(), prev_alpha_reciprocal, out=t_mat[k, k-1])
                t_mat[k-1, k].copy_(t_mat[k, k-1])

                if t_mat[k-1, k].max() < 1e-6:
                    update_tridiag = False
                    print("tridiag converges after {} iteration!".format(k+1))

            last_tridiag_iter = k

            prev_alpha_reciprocal.copy_(alpha_reciprocal)
            prev_beta.copy_(beta_tridiag)

    if use_rr:
        # NOTE: solution_sample1 and solution_sapmle2 is for backward pass computation
        if verbose:
            print("Original CG Residual after {} iterations: {}".format(k + 1, residual.norm()))

        if cg_solution_converged:
            print("\nUse cg solution for rr!")
            solution_sample1 = solution.narrow(-1, n_tridiag, rhs.size(-1) - n_tridiag)
            solution_sample2 = solution_sample1
        else:
            assert curr_Jvalue == max_iter, "curr_Jvalue = {}, max_iter = {}".format(curr_Jvalue, max_iter)

            while sorted_curr_Jidx < len(max_rrcg_iter_list) - 1:
                previous_original_curr_Jidx = sorted_max_rrct_iter_idx[sorted_curr_Jidx]
                sorted_curr_Jidx += 1

                original_curr_Jidx = sorted_max_rrct_iter_idx[sorted_curr_Jidx]
                rr_solution[original_curr_Jidx] = rr_solution[previous_original_curr_Jidx].clone()

            rr_solves = rr_solution.narrow(-1, n_tridiag, rhs.size(-1) - n_tridiag)
            solution = rr_solution.mean(0)

            half_rr_nsamples = int(rr_nsamples / 2)

            # solves without tridiag parts
            if half_rr_nsamples == 0:
                solution_sample1 = rr_solves[:half_rr_nsamples]  # empty size
            else:
                solution_sample1 = rr_solves[:half_rr_nsamples].mean(0)
            solution_sample2 = rr_solves[half_rr_nsamples:].mean(0)

    else:
        solution_sample1 = solution.narrow(-1, n_tridiag, rhs.size(-1) - n_tridiag)
        solution_sample2 = solution_sample1
        if verbose:
            print("Residual after {} iterations: {}".format(k+1, residual.norm()))

    rhs_norm_for_rr_solves = rhs_norm.narrow(-1, n_tridiag, rhs.size(-1) - n_tridiag)
    solution = solution.mul(rhs_norm)
    solution_sample1 = solution_sample1.mul(rhs_norm_for_rr_solves)
    solution_sample2 = solution_sample2.mul(rhs_norm_for_rr_solves)

    if not use_rr and not tolerance_reached and n_iter > 0:
        if n_tridiag:
            warnings.warn(
                "CG terminated in {} iterations with average residual norm {}."
                " The cg tolerance is {} specified by"
                " gpytorch.settings.cg_tolerance." 
                " Tridiag terminated in {} iterations with the last off-diagonal terms = {}." 
                " The tridiag tolerance is 1e-6."
                " If performance is affected, consider raising the maximum number of CG iterations by running code in"
                " a gpytorch.settings.max_cg_iterations(value) context.".format(
                    k + 1, residual_norm.mean(), tolerance,
                    last_tridiag_iter + 1, t_mat[last_tridiag_iter-1, last_tridiag_iter].max()),
                NumericalWarning,
            )
        else:
            warnings.warn(
                "CG terminated in {} iterations with average residual norm {}"
                " The cg tolerance is {} specified by"
                " gpytorch.settings.cg_tolerance."
                " If performance is affected, consider raising the maximum number of CG iterations by running code in"
                " a gpytorch.settings.max_cg_iterations(value) context.".format(
                    k + 1, residual_norm.mean(), tolerance,),
                NumericalWarning,
            )

    if is_vector:
        solution = solution.squeeze(-1)
    if n_tridiag:
        t_mat = t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1]
        return solution, solution_sample1, solution_sample2, \
               t_mat.permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous(), update_tridiag
    return solution, solution_sample1, solution_sample2


if __name__ == '__main__':
    # test

    torch.manual_seed(42)
    sig2 = 1.0
    lengthscale = 0.05
    #bsz = 200
    bsz = 10
    train_x = torch.linspace(-3, 3, bsz)
    train_y = torch.sin(train_x * (5 * math.pi)) * train_x + torch.randn_like(train_x) * 0.1
    train_y = train_y.unsqueeze(1)

    Kxx = matern(train_x, train_x, sig2, lengthscale, nu=1.5)

    # test cg
    print("solve using full cg")
    num_cg = bsz
    jitter = 1e-4
    solve_linear_cg, _, _ = linear_cg_rr(Kxx + jitter * torch.eye(bsz), train_y, max_iter=num_cg, use_rr=False,
                                   tolerance=1e-3, verbose=True, eps=1e-10)
    cg_residual = torch.matmul(Kxx, solve_linear_cg) - train_y
    print("cg residual = {}".format(cg_residual.norm()))

    # exact solve
    print("\nsolve using exact solve")
    solve_exact, _ = torch.solve(train_y, Kxx + jitter * torch.eye(bsz))  # (M, 1)
    residual = torch.matmul(Kxx, solve_exact) - train_y
    print("exact residaul = {}".format(residual.norm()))

    # test rr estimate
    print("\nsolving using rr")
    from rrcg.dist_of_iterations_for_rrcg import ExpDecayDist, UniformDist

    #N = 185
    N = 5
    dist = ExpDecayDist(temp=0.01, min=1, max=N)
    # dist = UniformDist(N=N)

    num_samples = 1
    rr_nsamples = 10 #1000
    #Jlist = dist.sample((rr_nsamples,))
    Jlist = torch.tensor([2, 1, 2, 3])
    print("J list = ", Jlist)
    solve_rr_list = torch.empty(num_samples, *train_y.shape)
    for i in range(num_samples):
        solve_linear_cg_rr, solve_linear_cg_rr_sample1, solve_linear_cg_rr_sapmle2 \
            = linear_cg_rr(Kxx + jitter * torch.eye(bsz), train_y, use_rr=True, dist_of_iter=dist,
                                          max_rrcg_iter_list=Jlist,
                                          tolerance=1e-3, eps=1e-10)
        solve_rr_list[i] = solve_linear_cg_rr
    residual_average = torch.matmul(Kxx, solve_rr_list.mean(dim=0)) - train_y
    print("\nrr Residual average = {}".format(residual_average.norm()))

    diff = solve_linear_cg - solve_rr_list.mean(dim=0)
    print(diff.norm())
