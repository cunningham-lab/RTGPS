import math

import torch
from gpytorch.distributions import Distribution
from gpytorch.lazy import LazyTensor, DiagLazyTensor
from torch import Tensor
import gpytorch
from gpytorch.module import Module
from gpytorch import settings, delazify, lazify
from gpytorch.utils.warnings import GPInputWarning, NumericalWarning
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.distributions import base_distributions
import warnings
from typing import Any

from torch.distributions import MultivariateNormal as TMultivariateNormal
from torch.distributions.utils import lazy_property, _standard_normal

from rrcg import rr_settings
from rrcg._inv_quad_log_det import inv_quad_logdet


class MultivariateNormal(TMultivariateNormal, Distribution):
    """
    Constructs a multivariate normal random variable, based on mean and covariance.
    Can be multivariate, or a batch of multivariate normals

    Passing a vector mean corresponds to a multivariate normal.
    Passing a matrix mean corresponds to a batch of multivariate normals.

    :param torch.tensor mean: Vector n or matrix b x n mean of mvn distribution.
    :param ~gpytorch.lazy.LazyTensor covar: Matrix n x n or batch matrix b x n x n covariance of
        mvn distribution.
    """

    def __init__(self, mean, covariance_matrix, validate_args=False):
        self._islazy = isinstance(mean, LazyTensor) or isinstance(covariance_matrix, LazyTensor)
        if self._islazy:
            if validate_args:
                ms = mean.size(-1)
                cs1 = covariance_matrix.size(-1)
                cs2 = covariance_matrix.size(-2)
                if not (ms == cs1 and ms == cs2):
                    raise ValueError(f"Wrong shapes in {self._repr_sizes(mean, covariance_matrix)}")
            self.loc = mean
            self._covar = covariance_matrix
            self.__unbroadcasted_scale_tril = None
            self._validate_args = validate_args
            batch_shape = _mul_broadcast_shape(self.loc.shape[:-1], covariance_matrix.shape[:-2])
            event_shape = self.loc.shape[-1:]
            # TODO: Integrate argument validation for LazyTensors into torch.distribution validation logic
            super(TMultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=False)
        else:
            super().__init__(loc=mean, covariance_matrix=covariance_matrix, validate_args=validate_args)

    @property
    def _unbroadcasted_scale_tril(self):
        if self.islazy and self.__unbroadcasted_scale_tril is None:
            # cache root decoposition
            ust = delazify(self.lazy_covariance_matrix.cholesky())
            self.__unbroadcasted_scale_tril = ust
        return self.__unbroadcasted_scale_tril

    @_unbroadcasted_scale_tril.setter
    def _unbroadcasted_scale_tril(self, ust):
        if self.islazy:
            raise NotImplementedError("Cannot set _unbroadcasted_scale_tril for lazy MVN distributions")
        else:
            self.__unbroadcasted_scale_tril = ust

    def add_jitter(self, noise=1e-4):
        return self.__class__(self.mean, self.lazy_covariance_matrix.add_jitter(noise))

    def expand(self, batch_size):
        new_loc = self.loc.expand(torch.Size(batch_size) + self.loc.shape[-1:])
        new_covar = self._covar.expand(torch.Size(batch_size) + self._covar.shape[-2:])
        res = self.__class__(new_loc, new_covar)
        return res

    def confidence_region(self):
        """
        Returns 2 standard deviations above and below the mean.

        :rtype: (torch.Tensor, torch.Tensor)
        :return: pair of tensors of size (b x d) or (d), where
            b is the batch size and d is the dimensionality of the random
            variable. The first (second) Tensor is the lower (upper) end of
            the confidence region.
        """
        std2 = self.stddev.mul_(2)
        mean = self.mean
        return mean.sub(std2), mean.add(std2)

    @staticmethod
    def _repr_sizes(mean, covariance_matrix):
        return f"MultivariateNormal(loc: {mean.size()}, scale: {covariance_matrix.size()})"

    @lazy_property
    def covariance_matrix(self):
        if self.islazy:
            return self._covar.evaluate()
        else:
            return super().covariance_matrix

    def get_base_samples(self, sample_shape=torch.Size()):
        """Get i.i.d. standard Normal samples (to be used with rsample(base_samples=base_samples))"""
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            base_samples = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return base_samples

    @lazy_property
    def lazy_covariance_matrix(self):
        """
        The covariance_matrix, represented as a LazyTensor
        """
        if self.islazy:
            return self._covar
        else:
            return lazify(super().covariance_matrix)

    def log_prob(self, value, dist_of_iter=None):
        if settings.fast_computations.log_prob.off():
            return super().log_prob(value)

        if self._validate_args:
            self._validate_sample(value)

        mean, covar = self.loc, self.lazy_covariance_matrix
        diff = value - mean

        # Repeat the covar to match the batch shape of diff
        if diff.shape[:-1] != covar.batch_shape:
            if len(diff.shape[:-1]) < len(covar.batch_shape):
                diff = diff.expand(covar.shape[:-1])
            else:
                padded_batch_shape = (*(1 for _ in range(diff.dim() + 1 - covar.dim())), *covar.batch_shape)
                covar = covar.repeat(
                    *(diff_size // covar_size for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)),
                    1,
                    1,
                )

        # Get log determininat and first part of quadratic form

        inv_quad, logdet = inv_quad_logdet(lazy_tsr=covar, inv_quad_rhs=diff.unsqueeze(-1), logdet=True,
                                           dist_of_iter=dist_of_iter)
        #inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)

        res = -0.5 * sum([inv_quad, logdet, diff.size(-1) * math.log(2 * math.pi)])
        return res

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        covar = self.lazy_covariance_matrix
        if base_samples is None:
            # Create some samples
            num_samples = sample_shape.numel() or 1

            # Get samples
            res = covar.zero_mean_mvn_samples(num_samples) + self.loc.unsqueeze(0)
            res = res.view(sample_shape + self.loc.shape)

        else:
            # Make sure that the base samples agree with the distribution
            if self.loc.shape != base_samples.shape[-self.loc.dim() :]:
                raise RuntimeError(
                    "The size of base_samples (minus sample shape dimensions) should agree with the size "
                    "of self.loc. Expected ...{} but got {}".format(self.loc.shape, base_samples.shape)
                )

            # Determine what the appropriate sample_shape parameter is
            sample_shape = base_samples.shape[: base_samples.dim() - self.loc.dim()]

            # Reshape samples to be batch_size x num_dim x num_samples
            # or num_bim x num_samples
            base_samples = base_samples.view(-1, *self.loc.shape)
            base_samples = base_samples.permute(*range(1, self.loc.dim() + 1), 0)

            # Now reparameterize those base samples
            covar_root = covar.root_decomposition().root
            # If necessary, adjust base_samples for rank of root decomposition
            if covar_root.shape[-1] < base_samples.shape[-2]:
                base_samples = base_samples[..., : covar_root.shape[-1], :]
            elif covar_root.shape[-1] > base_samples.shape[-2]:
                raise RuntimeError("Incompatible dimension of `base_samples`")
            res = covar_root.matmul(base_samples) + self.loc.unsqueeze(-1)

            # Permute and reshape new samples to be original size
            res = res.permute(-1, *range(self.loc.dim())).contiguous()
            res = res.view(sample_shape + self.loc.shape)

        return res

    def sample(self, sample_shape=torch.Size(), base_samples=None):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)

    def to_data_independent_dist(self):
        """
        Convert a MVN into a batched Normal distribution

        :returns: the bached data-independent Normal
        :rtype: gpytorch.distributions.Normal
        """
        # Create batch distribution where all data are independent, but the tasks are dependent
        try:
            # If pyro is installed, use that set of base distributions
            import pyro.distributions as base_distributions
        except ImportError:
            # Otherwise, use PyTorch
            import torch.distributions as base_distributions
        return base_distributions.Normal(self.mean, self.stddev)

    @property
    def stddev(self):
        # self.variance is guaranteed to be positive, because we do clamping.
        return self.variance.sqrt()

    @property
    def variance(self):
        if self.islazy:
            # overwrite this since torch MVN uses unbroadcasted_scale_tril for this
            diag = self.lazy_covariance_matrix.evaluate_kernel().diag()
            diag = diag.view(diag.shape[:-1] + self._event_shape)
            variance = diag.expand(self._batch_shape + self._event_shape)
        else:
            variance = super().variance

        # Check to make sure that variance isn't lower than minimum allowed value (default 1e-6).
        # This ensures that all variances are positive
        min_variance = settings.min_variance.value(variance.dtype)
        if variance.lt(min_variance).any():
            warnings.warn(
                f"Negative variance values detected. "
                "This is likely due to numerical instabilities. "
                f"Rounding negative variances up to {min_variance}.",
                NumericalWarning,
            )
            variance = variance.clamp_min(min_variance)
        return variance

    def __add__(self, other):
        if isinstance(other, MultivariateNormal):
            return self.__class__(
                mean=self.mean + other.mean,
                covariance_matrix=(self.lazy_covariance_matrix + other.lazy_covariance_matrix),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.mean + other, self.lazy_covariance_matrix)
        else:
            raise RuntimeError("Unsupported type {} for addition w/ MultivariateNormal".format(type(other)))

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        if not (isinstance(other, int) or isinstance(other, float)):
            raise RuntimeError("Can only multiply by scalars")
        if other == 1:
            return self
        return self.__class__(mean=self.mean * other, covariance_matrix=self.lazy_covariance_matrix * (other ** 2))

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        rest_idx = idx[:-1]
        last_idx = idx[-1]
        new_mean = self.mean[idx]

        if len(idx) <= self.mean.dim() - 1 and (Ellipsis not in rest_idx):
            new_cov = self.lazy_covariance_matrix[idx]
        elif len(idx) > self.mean.dim():
            raise IndexError(f"Index {idx} has too many dimensions")
        else:
            # In this case we know last_idx corresponds to the last dimension
            # of mean and the last two dimensions of lazy_covariance_matrix
            if isinstance(last_idx, int):
                new_cov = DiagLazyTensor(self.lazy_covariance_matrix.diag()[(*rest_idx, last_idx)])
            elif isinstance(last_idx, slice):
                new_cov = self.lazy_covariance_matrix[(*rest_idx, last_idx, last_idx)]
            elif last_idx is (...):
                new_cov = self.lazy_covariance_matrix[rest_idx]
            else:
                new_cov = self.lazy_covariance_matrix[(*rest_idx, last_idx, slice(None, None, None))][..., last_idx]
        return self.__class__(mean=new_mean, covariance_matrix=new_cov)


class GPRegressionModel(Module):
    def __init__(self, train_inputs, train_targets, likelihood, kernel_type='rbf', use_keops=True, ard_dim=None):
        super(GPRegressionModel, self).__init__()
        #self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()

        if torch.cuda.is_available() and use_keops:
            if kernel_type == 'rbf':
                kernel = gpytorch.kernels.keops.RBFKernel()
            elif kernel_type == 'matern05':
                kernel = gpytorch.kernels.keops.MaternKernel(nu=0.5)
            elif kernel_type == 'matern15':
                kernel = gpytorch.kernels.keops.MaternKernel(nu=1.5)
            elif kernel_type == 'matern25':
                kernel = gpytorch.kernels.keops.MaternKernel(nu=2.5)
            elif kernel_type == 'rbf-ard':
                if ard_dim is None:
                    ard_dim = train_inputs.shape[-1]
                    print(f"Getting ard_dim from training input, ard_dim = {ard_dim}")
                    kernel = gpytorch.kernels.keops.RBFKernel(ard_num_dims=ard_dim)
            else:
                raise ValueError("kernel_type must be chosen among {}, but got {}.".format(
                    ['rbf', 'matern05', 'matern15', 'matern25'], kernel_type))
        else:
            if kernel_type == 'rbf':
                kernel = gpytorch.kernels.RBFKernel()
            elif kernel_type == 'matern05':
                kernel = gpytorch.kernels.MaternKernel(nu=0.5)
            elif kernel_type == 'matern15':
                kernel = gpytorch.kernels.MaternKernel(nu=1.5)
            elif kernel_type == 'matern25':
                kernel = gpytorch.kernels.MaternKernel(nu=2.5)
            elif kernel_type == 'rbf-ard':
                if ard_dim is None:
                    ard_dim = train_inputs.shape[-1]
                    print(f"Getting ard_dim from training input, ard_dim = {ard_dim}")
                    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dim)
            else:
                raise ValueError("kernel_type must be chosen among {}, but got {}.".format(
                    ['rbf', 'matern05', 'matern15', 'matern25'], kernel_type))

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel,
        )

        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        if train_inputs is not None and not all(torch.is_tensor(train_input) for train_input in train_inputs):
            raise RuntimeError("Train inputs must be a tensor, or a list/tuple of tensors")

        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None

        self.likelihood = likelihood

        self.prediction_strategy = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.debug.on():
                if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    raise RuntimeError("You must train on the training inputs!")
            res = super().__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(GPRegressionModel, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            if settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super().__call__(*train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = _mul_broadcast_shape(batch_shape, train_input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = _mul_broadcast_shape(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training/test data
            full_output = super(GPRegressionModel, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            with settings._use_eval_tolerance():
                predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)


class GaussianLikelihood(Module):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size()):
        super().__init__()
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        self.noise_covar = noise_covar

    def __call__(self, input, *args, **kwargs):
        # Marginal
        assert isinstance(input, MultivariateNormal)
        return self.marginal(input, *args, **kwargs)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def log_marginal(
            self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        marginal = self.marginal(function_dist, *params, **kwargs)
        # We're making everything conditionally independent
        indep_dist = base_distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
        res = indep_dist.log_prob(observations)

        num_event_dim = len(function_dist.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(0, -num_event_dim, -1)))
        return res

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self.noise_covar(*params, shape=mean.shape, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)

    def log_marginal_from_marginal(
            self, marginal: MultivariateNormal, observations: Tensor) -> Tensor:
        indep_dist = base_distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
        res = indep_dist.log_prob(observations)

        num_event_dim = len(marginal.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(0, -num_event_dim, -1)))

        num_event_dim = len(marginal.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(0, -num_event_dim, -1)))
        return res


class MLL(Module):
    def __init__(self, likelihood, model):
        super().__init__()
        self.likelihood = likelihood
        self.model = model

    def forward(self, function_dist, target, *params, dist_of_iter=None):
        output = self.likelihood(function_dist, *params)

        res = output.log_prob(target, dist_of_iter)
        # res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = target.size(-1)
        return res.div_(num_data)


class CholeskyModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        #self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_inv_quad_logdet(model, train_x, train_y, method='cholesky', num_cg=None, dist_of_iter=False,
                        rr_sample_num=1, use_gpytorch=False):

    with settings.prior_mode():
        output = model(train_x)  # Multivariate Normal
    marginal_output = model.likelihood(output, train_y, dist_of_iter=dist_of_iter)

    mean, covar = marginal_output.loc, marginal_output.lazy_covariance_matrix
    diff = train_y - mean
    # Repeat the covar to match the batch shape of diff
    if diff.shape[:-1] != covar.batch_shape:
        if len(diff.shape[:-1]) < len(covar.batch_shape):
            diff = diff.expand(covar.shape[:-1])
        else:
            padded_batch_shape = (*(1 for _ in range(diff.dim() + 1 - covar.dim())), *covar.batch_shape)
            covar = covar.repeat(
                *(diff_size // covar_size for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)),
                1,
                1,
            )

    from rrcg._inv_quad_log_det import inv_quad_logdet
    # Get log determininat and first part of quadratic form

    with torch.no_grad():
        if method == 'cholesky':
            with gpytorch.settings.max_cholesky_size(1000000):
                inv_quad, logdet = inv_quad_logdet(lazy_tsr=covar, inv_quad_rhs=diff.unsqueeze(-1), logdet=True,
                                                   dist_of_iter=dist_of_iter)
                return inv_quad.item(), logdet.item()
        with gpytorch.settings.max_cholesky_size(0):
            with gpytorch.settings.cg_tolerance(1e-50):
                if method == 'rrcg':
                    inv_quad_list = []
                    logdet_list = []
                    with rr_settings.use_rr_cg():
                        with rr_settings.use_rr_lanczos():
                            for i in range(rr_sample_num):
                                inv_quad, logdet = inv_quad_logdet(lazy_tsr=covar, inv_quad_rhs=diff.unsqueeze(-1), logdet=True,
                                                       dist_of_iter=dist_of_iter)
                                inv_quad_list.append(inv_quad.item())
                                logdet_list.append(logdet.item())
                    if rr_sample_num == 1:
                        return inv_quad_list[0], logdet_list[0]
                    return inv_quad_list, logdet_list

                # otherwise, use cg
                assert method == 'cg', method
                assert num_cg is not None
                with gpytorch.settings.max_cg_iterations(num_cg):
                    with gpytorch.settings.max_lanczos_quadrature_iterations(num_cg):
                        if use_gpytorch:
                            return covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)
                        inv_quad, logdet = inv_quad_logdet(lazy_tsr=covar, inv_quad_rhs=diff.unsqueeze(-1), logdet=True,
                                                           dist_of_iter=dist_of_iter)
                        return inv_quad.item(), logdet.item()


