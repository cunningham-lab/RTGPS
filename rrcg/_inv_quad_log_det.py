#!/usr/bin/env python3

import warnings

import torch
from torch.autograd import Function

from rrcg.linear_cg import linear_cg_rr
from rrcg import rr_settings

from gpytorch import settings
from gpytorch.utils.lanczos import lanczos_tridiag_to_diag
from gpytorch.utils.stochastic_lq import StochasticLQ
from gpytorch.lazy.lazy_tensor import LazyTensor


class InvQuadLogDet(Function):
    """
    Given a PSD matrix A (or a batch of PSD matrices A), this function computes one or both
    of the following
    - The matrix solves A^{-1} b
    - logdet(A)
    """

    @staticmethod
    def forward(
            ctx,
            representation_tree,
            dtype,
            device,
            matrix_shape,
            batch_shape=torch.Size(),
            inv_quad=False,
            logdet=False,
            probe_vectors=None,
            probe_vector_norms=None,
            dist_of_iter=None,
            *args,
    ):
        """
        *args - The arguments representing the PSD matrix A (or batch of PSD matrices A)
        If self.inv_quad is true, the first entry in *args is inv_quad_rhs (Tensor)
        - the RHS of the matrix solves.

        Returns:
        - (Scalar) The inverse quadratic form (or None, if self.inv_quad is False)
        - (Scalar) The log determinant (or None, self.if logdet is False)
        """

        if not (inv_quad or logdet):
            raise RuntimeError("Either inv_quad or logdet must be true (or both)")

        ctx.representation_tree = representation_tree
        ctx.dtype = dtype
        ctx.device = device
        ctx.matrix_shape = matrix_shape
        ctx.batch_shape = batch_shape
        ctx.inv_quad = inv_quad
        ctx.logdet = logdet

        inv_quad_rhs = None
        if ctx.inv_quad:
            matrix_args = args[1:]
            inv_quad_rhs = args[0]
        else:
            matrix_args = args

        # Get closure for matmul
        lazy_tsr = ctx.representation_tree(*matrix_args)
        with torch.no_grad():
            preconditioner, precond_lt, logdet_correction = lazy_tsr._preconditioner()

        ctx.preconditioner = preconditioner

        if (probe_vectors is None or probe_vector_norms is None) and logdet:
            num_random_probes = settings.num_trace_samples.value()
            if preconditioner is None:
                if settings.deterministic_probes.on():
                    warnings.warn(
                        "Deterministic probes will currently work only if you aren't training multiple independent"
                        " models simultaneously.",
                        UserWarning,
                    )
                    if settings.deterministic_probes.probe_vectors is None:
                        probe_vectors = torch.empty(matrix_shape[-1], num_random_probes, dtype=dtype, device=device)
                        probe_vectors.bernoulli_().mul_(2).add_(-1)
                        settings.deterministic_probes.probe_vectors = probe_vectors
                    else:
                        probe_vectors = settings.deterministic_probes.probe_vectors
                else:
                    probe_vectors = torch.empty(matrix_shape[-1], num_random_probes, dtype=dtype, device=device)
                    probe_vectors.bernoulli_().mul_(2).add_(-1)

                probe_vector_norms = torch.norm(probe_vectors, 2, dim=-2, keepdim=True)
                if batch_shape is not None:
                    probe_vectors = probe_vectors.expand(*batch_shape, matrix_shape[-1], num_random_probes)
                    probe_vector_norms = probe_vector_norms.expand(*batch_shape, 1, num_random_probes)
            else:  # When preconditioning, probe vectors must be drawn from N(0, P)
                if precond_lt.size()[-2:] == torch.Size([1, 1]):
                    covar_root = precond_lt.evaluate().sqrt()
                else:
                    covar_root = precond_lt.root_decomposition().root

                if settings.deterministic_probes.on():
                    warnings.warn(
                        "Deterministic probes will currently work only if you aren't training multiple independent"
                        " models simultaneously.",
                        UserWarning,
                    )
                    base_samples = settings.deterministic_probes.probe_vectors
                    if base_samples is None or covar_root.size(-1) != base_samples.size(-2):
                        base_samples = torch.randn(
                            *precond_lt.batch_shape,
                            covar_root.size(-1),
                            num_random_probes,
                            dtype=precond_lt.dtype,
                            device=precond_lt.device,
                        )
                        settings.deterministic_probes.probe_vectors = base_samples

                    probe_vectors = covar_root.matmul(base_samples).permute(-1, *range(precond_lt.dim() - 1))
                else:
                    base_samples = torch.randn(
                        *precond_lt.batch_shape,
                        covar_root.size(-1),
                        num_random_probes,
                        dtype=precond_lt.dtype,
                        device=precond_lt.device,
                    )
                    probe_vectors = precond_lt.zero_mean_mvn_samples(num_random_probes)
                probe_vectors = probe_vectors.unsqueeze(-2).transpose(0, -2).squeeze(0).transpose(-2, -1).contiguous()
                probe_vector_norms = torch.norm(probe_vectors, p=2, dim=-2, keepdim=True)
            probe_vectors = probe_vectors.div(probe_vector_norms)

        ctx.probe_vectors = probe_vectors
        ctx.probe_vector_norms = probe_vector_norms

        if ctx.logdet and not ctx.probe_vectors.numel():
            raise RuntimeError("Probe vectors were not supplied for logdet computation")

        # Collect terms for LinearCG
        # We use LinearCG for both matrix solves and for stochastically estimating the log det
        rhs_list = []
        num_random_probes = 0
        num_inv_quad_solves = 0

        # RHS for logdet
        if ctx.logdet:
            rhs_list.append(ctx.probe_vectors)
            num_random_probes = ctx.probe_vectors.size(-1)

        # RHS for inv_quad
        ctx.is_vector = False
        if ctx.inv_quad:
            if inv_quad_rhs.ndimension() == 1:
                inv_quad_rhs = inv_quad_rhs.unsqueeze(-1)
                ctx.is_vector = True
            rhs_list.append(inv_quad_rhs)
            num_inv_quad_solves = inv_quad_rhs.size(-1)

        # Perform solves (for inv_quad) and tridiagonalization (for estimating logdet)
        rhs = torch.cat(rhs_list, -1)
        t_mat = None

        # rr settings
        use_rr_for_cg = rr_settings.use_rr_cg.on()
        use_rr_lanczos = rr_settings.use_rr_lanczos.on() # this is only enabled when use_rr_for_cg
        max_rrcg_iter_list = None
        max_tridiag_iter = None
        if use_rr_for_cg:
            if rr_settings.use_prespecified_rr_iter.on():
                max_rrcg_iter_list = rr_settings.max_rr_cg_iter_list.value()
                max_rrcg_iter_list = torch.tensor(max_rrcg_iter_list, dtype=torch.int)
                rr_nsamples = max_rrcg_iter_list.shape[0]
            else:
                rr_nsamples = rr_settings.rr_cg_nsamples.value()
                max_rrcg_iter_list = dist_of_iter.sample((rr_nsamples,))

            if not use_rr_lanczos:
                # we just use a random tridiag iter, and so in average tridiag_iter = rrdist.mean
                max_tridiag_iter = max_rrcg_iter_list[0]
            # else, max_tridiag_iter will be set to max_iter in linear_cg_rr

        if ctx.logdet and settings.skip_logdet_forward.off():
            solves, inv_quad_solves_sample1, inv_quad_solves_sample2, t_mat, update_tridiag = \
                linear_cg_rr(matmul_closure=lazy_tsr._matmul, rhs=rhs,
                             use_rr=use_rr_for_cg, max_iter=None, max_rrcg_iter_list=max_rrcg_iter_list,
                             max_tridiag_iter=max_tridiag_iter, dist_of_iter=dist_of_iter,
                             n_tridiag=num_random_probes, preconditioner=preconditioner)
        else:
            max_iter = settings.max_cg_iterations.value()
            solves, inv_quad_solves_sample1, inv_quad_solves_sample2 = \
                linear_cg_rr(matmul_closure=lazy_tsr._matmul, rhs=rhs,
                             use_rr=use_rr_for_cg, max_iter=max_iter, max_rrcg_iter_list=max_rrcg_iter_list,
                             dist_of_iter=dist_of_iter,
                             n_tridiag=0, preconditioner=preconditioner)

        # Final values to return
        logdet_term = torch.zeros(lazy_tsr.batch_shape, dtype=ctx.dtype, device=ctx.device)
        inv_quad_term = torch.zeros(lazy_tsr.batch_shape, dtype=ctx.dtype, device=ctx.device)

        # Compute logdet from tridiagonalization
        if ctx.logdet and settings.skip_logdet_forward.off():
            if torch.any(torch.isnan(t_mat)).item():
                logdet_term = torch.tensor(float("nan"), dtype=ctx.dtype, device=ctx.device)
            else:
                if ctx.batch_shape is None:
                    t_mat = t_mat.unsqueeze(1)
                if use_rr_lanczos and update_tridiag:
                    # only use_rr to estimate logdet term when lanczos process is not converged
                    # t_mat: (num_probes, J, J)

                    slq = StochasticLQ()
                    max_iter = t_mat.shape[-1]

                    sorted_max_rrcg_iter_list, _ = torch.sort(max_rrcg_iter_list)
                    assert max_iter == sorted_max_rrcg_iter_list[-1], \
                        "t_mat shape = {}, but sorted_max_rrcg_iter_list = {}".format(t_mat.shape,
                                                                                      sorted_max_rrcg_iter_list)

                    # TODO: test lazy_tsr_batchshape for rr logdet_term
                    logdet_term = torch.zeros((rr_nsamples,) + lazy_tsr.batch_shape, dtype=t_mat.dtype,
                                              device=t_mat.device)

                    curr_Jidx = 0
                    curr_Jvalue = sorted_max_rrcg_iter_list[0]

                    # compute the first term
                    t_mat_k = t_mat[:, :1, :1]
                    eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat_k)
                    (logdet_term_k,) = slq.evaluate(ctx.matrix_shape, eigenvalues, eigenvectors, [lambda x: x.log()])
                    torch.add(logdet_term[curr_Jidx], logdet_term_k, out=logdet_term[curr_Jidx])

                    logdet_term_km1 = logdet_term_k

                    for k in range(1, max_iter):
                        # compute #rr_nsamples estimates of logdet terms
                        while k + 1 > curr_Jvalue:
                            # move to next rr-estimate, and update curr_Jidx
                            curr_Jidx += 1
                            curr_Jvalue = sorted_max_rrcg_iter_list[curr_Jidx]
                            # we need to copy accumulated summation from previous
                            logdet_term[curr_Jidx] = logdet_term[curr_Jidx - 1].clone()

                        t_mat_k = t_mat[:, :k + 1, :k + 1]
                        eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat_k)
                        (logdet_term_k,) = slq.evaluate(ctx.matrix_shape, eigenvalues, eigenvectors,
                                                        [lambda x: x.log()])
                        logdet_term_diff = logdet_term_k - logdet_term_km1

                        prob_k = (1 - dist_of_iter.cdf(k))
                        torch.div(logdet_term_diff, prob_k, out=logdet_term_diff)
                        torch.add(logdet_term[curr_Jidx], logdet_term_diff, out=logdet_term[curr_Jidx])

                        logdet_term_km1 = logdet_term_k

                    while curr_Jidx < rr_nsamples - 1:
                        curr_Jidx += 1
                        logdet_term[curr_Jidx] = logdet_term[curr_Jidx - 1].clone()

                    logdet_term = logdet_term.mean()

                else:
                    eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat)  # (num_probes, J), (num_probes, J, J)
                    slq = StochasticLQ()
                    (logdet_term,) = slq.evaluate(ctx.matrix_shape, eigenvalues, eigenvectors,
                                                  [lambda x: x.log()])  # scalar value

                # Add correction
                if logdet_correction is not None:
                    logdet_term = logdet_term + logdet_correction

        # Extract inv_quad solves from all solves
        if ctx.inv_quad:
            inv_quad_solves = solves.narrow(-1, num_random_probes, num_inv_quad_solves)
            inv_quad_term = (inv_quad_solves * inv_quad_rhs).sum(-2)

        ctx.num_random_probes = num_random_probes
        ctx.num_inv_quad_solves = num_inv_quad_solves

        if ctx.inv_quad:
            probe_vector_solves = solves.narrow(-1, 0, num_random_probes)
            to_save = list(matrix_args) + [probe_vector_solves, inv_quad_solves_sample1, inv_quad_solves_sample2]
        else:
            # only logdet is required, in this case, solves = probe_vector_solves
            to_save = list(matrix_args) + [solves]
        ctx.save_for_backward(*to_save)

        if settings.memory_efficient.off():
            ctx._lazy_tsr = lazy_tsr
        return inv_quad_term, logdet_term

    @staticmethod
    def backward(ctx, inv_quad_grad_output, logdet_grad_output):
        matrix_arg_grads = None
        inv_quad_rhs_grad = None

        # Which backward passes should we compute?
        compute_inv_quad_grad = inv_quad_grad_output.abs().sum() and ctx.inv_quad
        compute_logdet_grad = logdet_grad_output.abs().sum() and ctx.logdet

        # Get input arguments, and get gradients in the proper form
        # matrix_args = ctx.saved_tensors[:-1]
        # solves = ctx.saved_tensors[-1]

        if ctx.inv_quad:
            matrix_args = ctx.saved_tensors[:-3]
            probe_vector_solves = ctx.saved_tensors[-3]
            inv_quad_solves_sample1 = ctx.saved_tensors[-2]
            if inv_quad_solves_sample1.size()[0] == 0:
                raise ValueError("inv_quad_solves_sample1 should not be empty,"
                                 " but got inv_quad_solves_sample1.size = {}."
                                 " If using rr, you should specify the value of rr_setings.rr_nsamples to be >= 2".format(
                    inv_quad_solves_sample1.size()))
            inv_quad_solves_sample2 = ctx.saved_tensors[-1]

            inv_quad_grad_output = inv_quad_grad_output.unsqueeze(-2)
        else:
            # only logdet is required
            matrix_args = ctx.saved_tensors[:-1]
            probe_vector_solves = ctx.saved_tensors[-1]

        if hasattr(ctx, "_lazy_tsr"):
            lazy_tsr = ctx._lazy_tsr
        else:
            lazy_tsr = ctx.representation_tree(*matrix_args)

        if compute_logdet_grad:
            logdet_grad_output = logdet_grad_output.unsqueeze(-1)
            logdet_grad_output.unsqueeze_(-1)

        # Divide up the solves
        if compute_logdet_grad:
            coef = 1.0 / ctx.probe_vectors.size(-1)
            probe_vector_solves = probe_vector_solves.mul(coef)
            probe_vector_solves.mul_(ctx.probe_vector_norms).mul_(logdet_grad_output)
            probe_vectors = ctx.probe_vectors.mul(ctx.probe_vector_norms)
        if ctx.inv_quad:
            neg_inv_quad_solves_sample1_times_grad_out = inv_quad_solves_sample1.mul(inv_quad_grad_output).mul_(-1)

        # input_1 gradient
        if any(ctx.needs_input_grad):
            # Collect terms for arg grads
            left_factors_list = []
            right_factors_list = []

            if compute_logdet_grad:
                left_factors_list.append(probe_vector_solves)
                if ctx.preconditioner is not None:
                    probe_vectors = ctx.preconditioner(probe_vectors)
                right_factors_list.append(probe_vectors)

            if compute_inv_quad_grad:
                # left_factors_list.append(neg_inv_quad_solves_times_grad_out)
                # right_factors_list.append(inv_quad_solves)
                left_factors_list.append(neg_inv_quad_solves_sample1_times_grad_out)
                right_factors_list.append(inv_quad_solves_sample2)

            left_factors = torch.cat(left_factors_list, -1)
            right_factors = torch.cat(right_factors_list, -1)
            matrix_arg_grads = lazy_tsr._quad_form_derivative(left_factors, right_factors)

        # input_2 gradients
        if compute_inv_quad_grad and ctx.needs_input_grad[10]:
            # inv_quad_rhs_grad = neg_inv_quad_solves_times_grad_out.mul_(-2)
            # TODO: instead of using sample1, we can use full sample
            inv_quad_rhs_grad = neg_inv_quad_solves_sample1_times_grad_out.mul_(-2)
        elif ctx.inv_quad:
            # inv_quad_rhs_grad = torch.zeros_like(inv_quad_solves)
            inv_quad_rhs_grad = torch.zeros_like(inv_quad_solves_sample1)
        if ctx.is_vector:
            inv_quad_rhs_grad.squeeze_(-1)

        if ctx.inv_quad:
            res = [inv_quad_rhs_grad] + list(matrix_arg_grads)
        else:
            res = list(matrix_arg_grads)

        return tuple([None] * 10 + res)


def inv_quad_logdet(lazy_tsr, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True, dist_of_iter=None):
    assert isinstance(lazy_tsr, LazyTensor)

    # Special case: use Cholesky to compute these terms
    if settings.fast_computations.log_prob.off() or (lazy_tsr.size(-1) <= settings.max_cholesky_size.value()):
        from gpytorch.lazy.chol_lazy_tensor import CholLazyTensor
        from gpytorch.lazy.triangular_lazy_tensor import TriangularLazyTensor

        cholesky = CholLazyTensor(TriangularLazyTensor(lazy_tsr.cholesky()))
        return cholesky.inv_quad_logdet(inv_quad_rhs=inv_quad_rhs, logdet=logdet, reduce_inv_quad=reduce_inv_quad)

    # Default: use modified batch conjugate gradients to compute these terms
    # See NeurIPS 2018 paper: https://arxiv.org/abs/1809.11165
    if not lazy_tsr.is_square:
        raise RuntimeError(
            "inv_quad_logdet only operates on (batches of) square (positive semi-definite) LazyTensors. "
            "Got a {} of size {}.".format(lazy_tsr.__class__.__name__, lazy_tsr.size())
        )

    if inv_quad_rhs is not None:
        if lazy_tsr.dim() == 2 and inv_quad_rhs.dim() == 1:
            if lazy_tsr.shape[-1] != inv_quad_rhs.numel():
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        lazy_tsr.shape, inv_quad_rhs.shape
                    )
                )
        elif lazy_tsr.dim() != inv_quad_rhs.dim():
            raise RuntimeError(
                "LazyTensor (size={}) and right-hand-side Tensor (size={}) should have the same number "
                "of dimensions.".format(lazy_tsr.shape, inv_quad_rhs.shape)
            )
        elif lazy_tsr.batch_shape != inv_quad_rhs.shape[:-2] or lazy_tsr.shape[-1] != inv_quad_rhs.shape[-2]:
            raise RuntimeError(
                "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                    lazy_tsr.shape, inv_quad_rhs.shape
                )
            )

    args = lazy_tsr.representation()  # TODO: check this

    if inv_quad_rhs is not None:
        args = [inv_quad_rhs] + list(args)

    probe_vectors, probe_vector_norms = lazy_tsr._probe_vectors_and_norms()

    func = InvQuadLogDet.apply

    inv_quad_term, logdet_term = func(
        lazy_tsr.representation_tree(),
        lazy_tsr.dtype,
        lazy_tsr.device,
        lazy_tsr.matrix_shape,
        lazy_tsr.batch_shape,
        (inv_quad_rhs is not None),
        logdet,
        probe_vectors,
        probe_vector_norms,
        dist_of_iter,
        *args,
    )

    if inv_quad_term.numel() and reduce_inv_quad:
        inv_quad_term = inv_quad_term.sum(-1)
    return inv_quad_term, logdet_term

