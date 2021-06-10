# %% md

# GPs with Cholesky, CG, and RFFs

# %%

import math
import torch
import warnings
import gpytorch
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import numpy as np




# %% md

### Set up training/evaluation data

# %%

train_x = torch.linspace(0, 1, 200)
train_y = torch.sin(train_x * (5 * math.pi)) * train_x + torch.randn_like(train_x) * 0.1
test_x = torch.linspace(-1, 2, 151)


# %% md

### Model for Cholesky/CG

# %%

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# %%

# help(gpytorch.kernels.ScaleKernel)

# %% md

### Model for RFFs (uses a different kernel)

# %%

class GPRegressionModelRFF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_rff_samples):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RFFKernel(num_samples=num_rff_samples),  # This is the different line!
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# %% md

### Training/testing scripts

# %%

def train(model, likelihood, name=""):
    likelihood.initialize(noise=0.01)

    optimizer = torch.optim.Adam(
        model.covar_module.base_kernel.parameters(), lr=0.05)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    iterator = tqdm(range(500), desc=f"{name} Training")

    for i in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        iterator.set_postfix(
            loss=loss.item(), ls=model.covar_module.base_kernel.lengthscale.item(),
            os=model.covar_module.outputscale.item(), noise=likelihood.noise.item()
        )

        # Reset RFF weights
        if hasattr(model.covar_module.base_kernel, "randn_weights"):
            model.covar_module.base_kernel.randn_weights.normal_()


# %%

def eval(model, likelihood, name=""):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(test_x))
        mean = prediction.mean
        lower, upper = prediction.confidence_region()
        ax.plot(train_x.numpy(), train_y.numpy(), "k*", label="Training Data")
        ax.plot(test_x.numpy(), mean.numpy(), "b", label="Prediction")
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set(xlabel="x", ylabel="y", title=name, ylim=(-6., 6.))

    return fig





# %%


# %% md

## Cholesky training/evaluation

# %%

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

# Force GPyTorch to use Cholesky
with gpytorch.settings.max_cholesky_size(10000000):
    train(model, likelihood, name="Cholesky")
    eval(model, likelihood, name="Cholesky")


# %% md

# CG training/evaluation

# %%

def CG_training(model, likelihood, num_cg):
    '''mostly setting up the context managers to force GPyTorch to use CG and not Cholesky,
    and to allow for a small number of them. assuming that train is
    defined outside of this function (change in future).
    GPs with kernel matrices that are smaller than max_cholesky_size use Cholesky for inference.
    If the matrix is bigger than max_cholesky_size, then GPyTorch uses CG for inference.
    By setting max_cholesky_size(10000000000),
    we’re forcing GPyTorch to use Cholesky for evaluation (after using CG for kernel learning). It doesn’t make a huge difference in this scenario.'''

    # Force GPyTorch to use CG
    # We'll also hard-code the number of iterations
    with gpytorch.settings.max_cholesky_size(0):
        # Silence CG-related warnings
        warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)

        # Since the tolerance is impossibly low, GPyTorch will be
        # forced to use the `max_cg_iterations` stopping criterion
        # We'll also not use the preconditioner
        with gpytorch.settings.max_cg_iterations(num_cg), gpytorch.settings.max_lanczos_quadrature_iterations(num_cg):
            with gpytorch.settings.cg_tolerance(1e-50):
                # To make the effects of early termination more dramatic, we'll turn off the preconditioner
                with gpytorch.settings.max_preconditioner_size(0):  # Default is usually 10
                    train(model, likelihood, name="CG")


# %%

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)
num_cg = 5

#from gpytorch.utils.linear_cg import linear_cg
#linear_cg()

# train
CG_training(model, likelihood, num_cg)  # context managing around train()
# eval
with gpytorch.settings.max_cholesky_size(100000000):
    eval(model, likelihood, name="CG")

"""
# %%

# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = GPRegressionModel(train_x, train_y, likelihood)

# # Force GPyTorch to use CG
# # We'll also hard-code the number of iterations
# with gpytorch.settings.max_cholesky_size(0):

#     # Silence CG-related warnings
#     warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)

#     # Since the tolerance is impossibly low, GPyTorch will be
#     # forced to use the `max_cg_iterations` stopping criterion
#     # We'll also not use the preconditioner
#     num_cg = 5
#     with gpytorch.settings.max_cg_iterations(num_cg), gpytorch.settings.max_lanczos_quadrature_iterations(num_cg):
#         with gpytorch.settings.cg_tolerance(1e-50):

#             # To make the effects of early termination more dramatic, we'll turn off the preconditioner
#             with gpytorch.settings.max_preconditioner_size(0):  # Default is usually 10
#                 train(model, likelihood, name="CG")

# # We can use a more strict CG convergence criterion for evaluation
# with gpytorch.settings.max_cholesky_size(100000000):
#     #model.train()
#     eval(model, likelihood, name="CG")

# %% md

## RFF training/evaluation

# %%

likelihood = gpytorch.likelihoods.GaussianLikelihood()
# Control the number of RFF samples here!
model = GPRegressionModelRFF(train_x, train_y, likelihood, num_rff_samples=20)

# Force GPyTorch to use Cholesky
with gpytorch.settings.max_cholesky_size(10000000):
    train(model, likelihood, name="RFFs")
    eval(model, likelihood, name="RFFs")

# %% md

### sample from prior and posterior
In
this
section
I
show
how
to
sample
from the prior and the
posterior,
without
training.

# %%

# initialize instances and set params
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

'''see basic usage tutorial on 
https://docs.gpytorch.ai/en/v1.2.0/examples/00_Basic_Usage/Hyperparameters.html'''
hypers = {
    'likelihood.noise_covar.noise': torch.tensor(.001),
    'covar_module.base_kernel.lengthscale': torch.tensor(.3),
    'covar_module.outputscale': torch.tensor(2.),
}

model.initialize(**hypers)
print(
    model.likelihood.noise_covar.noise.item(),
    model.covar_module.base_kernel.lengthscale.item(),
    model.covar_module.outputscale.item()
)
print('showing named parameters:')
print(list(model.named_parameters()))


# %%

def sample_from_prior(model, likelihood, train_x):
    # https://github.com/cornellius-gp/gpytorch/pull/707
    with gpytorch.settings.prior_mode(True):
        prior_preds = likelihood(model(train_x))
    return prior_preds.sample()


# %%

# showing how to take a sample from the prior
torch.manual_seed(0)
prior_sample = sample_from_prior(model, likelihood, train_x)
plt.plot(train_x, prior_sample.detach().numpy(), 'x')
plt.title('Sample from the prior');
plt.xlabel(r'$x$');
plt.ylabel(r'$y = f(x) + \epsilon$');
plt.ylim([-5, 5])

# %%

# Set into posterior mode and take a sample (no training)
model.eval()
likelihood.eval()

post_preds = likelihood(model(train_x))  # post predictive distribution
post_sample = post_preds.sample()  # sample from it

# %%

plt.subplot(141)
plt.title('prior cov')
plt.imshow(prior_preds.covariance_matrix.detach().numpy())
plt.subplot(142)
plt.title('posterior cov')
plt.imshow(post_preds.covariance_matrix.detach().numpy())
plt.subplot(143)
plt.title('post - prior')
plt.imshow(post_preds.covariance_matrix.detach().numpy() -
           prior_preds.covariance_matrix.detach().numpy())
plt.subplot(144)
plt.scatter(prior_preds.loc.detach().numpy(),
            post_preds.loc.detach().numpy())
plt.xlabel('prior mean')
plt.ylabel('post mean')
plt.tight_layout()

# %% md

### Experiments: varying RFF and CG samples

# %%

# np.concatenate([np.array(10), np.arange(20,31,10)])
np.linspace(20, 30, 10)
np.concatenate((np.array([10, 50]), np.arange(20, 30, 10)))

# %%

# parameters of the experiment
lengthscale_vec = np.linspace(0.1, 2, 11)
RFF_samples = np.concatenate((np.array([10, 50]), np.arange(100, 1000, 100), np.arange(1000, 3000, 250)))
CG_iters = np.arange(4, 54, 4)
print('experimental params:')
print('lengthscales:')
print(lengthscale_vec)
print('RFF samples:')
print(RFF_samples)
print('CG_iters:')
print(CG_iters)

# %%

train_x = torch.linspace(-3, 3, 200)
train_y = torch.sin(train_x * (5 * math.pi)) * train_x + torch.randn_like(train_x) * 0.1
test_x = torch.linspace(-4, 5, 151)

# %%

# for self: if you want to randomize initial values. 
# torch.clamp(torch.tensor(2.5) + 0.5*torch.randn(1), 0.05)

# %%

import time

torch.manual_seed(0)

recovered_lengthscales_RFF = np.zeros((len(lengthscale_vec),
                                       len(RFF_samples)))

recovered_lengthscales_CG = np.zeros((len(lengthscale_vec),
                                      len(CG_iters)))

training_time_RFF = np.zeros_like(recovered_lengthscales_RFF)
training_time_CG = np.zeros_like(recovered_lengthscales_CG)

rand_init_lengthscales_RFF = (lengthscale_vec[0] - lengthscale_vec[-1]) * \
                             torch.rand(len(lengthscale_vec)) + lengthscale_vec[-1]
rand_init_lengthscales_CG = (lengthscale_vec[0] - lengthscale_vec[-1]) * \
                            torch.rand(len(lengthscale_vec)) + lengthscale_vec[-1]

for i in range(len(lengthscale_vec)):

    # re-initialize true model
    true_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    true_model = GPRegressionModel(train_x, train_y, true_likelihood)

    '''see basic usage tutorial on 
    https://docs.gpytorch.ai/en/v1.2.0/examples/00_Basic_Usage/Hyperparameters.html'''
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(.005),
        'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale_vec[i]),
        'covar_module.outputscale': torch.tensor(2.),
    }

    true_model.initialize(**hypers)

    print("true lengthscale= %.2f" % true_model.covar_module.base_kernel.lengthscale.item())
    #     print(
    #         true_model.likelihood.noise_covar.noise.item(),
    #         true_model.covar_module.base_kernel.lengthscale.item(),
    #         true_model.covar_module.outputscale.item()
    #     )

    train_y = sample_from_prior(true_model, true_likelihood, train_x)
    plt.plot(train_y, '*')
    plt.show()

    for j in range(len(RFF_samples)):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModelRFF(train_x, train_y,
                                     likelihood,
                                     num_rff_samples=RFF_samples[j])
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(.01),
            'covar_module.base_kernel.lengthscale': rand_init_lengthscales_RFF[i],
            'covar_module.outputscale': torch.tensor(2.),
        }

        model.initialize(**hypers)

        print('pre training: likelihood noise; kernel lengthscale; outputscale')
        print(
            model.likelihood.noise_covar.noise.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.covar_module.outputscale.item()
        )  # model.covar_module.base_kernel.lengthscale = rand_init_lengthscales_RFF[j]
        print(model.covar_module.base_kernel.lengthscale.item())

        with gpytorch.settings.max_cholesky_size(10000000):
            start = time.time()
            train(model, likelihood, name="RFFs")
            train_time = time.time() - start
            eval(model, likelihood, name="RFFs")
            plt.show()

        print('post training lengthscale')
        print(model.covar_module.base_kernel.lengthscale.item())

        recovered_lengthscales_RFF[i, j] = model.covar_module.base_kernel.lengthscale.item()
        training_time_RFF[i, j] = train_time

    for c in range(len(CG_iters)):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        CG_model = GPRegressionModel(train_x, train_y, likelihood)

        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(.01),
            'covar_module.base_kernel.lengthscale': rand_init_lengthscales_CG[i],
            'covar_module.outputscale': torch.tensor(2.),
        }

        CG_model.initialize(**hypers)
        print('pre training: likelihood noise; kernel lengthscale; outputscale')
        print(
            CG_model.likelihood.noise_covar.noise.item(),
            CG_model.covar_module.base_kernel.lengthscale.item(),
            CG_model.covar_module.outputscale.item()
        )  # model.covar_module.base_kernel.lengthscale = rand_init_lengthscales_RFF[j]
        print(model.covar_module.base_kernel.lengthscale.item())
        # train
        start = time.time()
        CG_training(CG_model, likelihood, CG_iters[c])  # context managing around train()
        train_time = time.time() - start
        # eval
        with gpytorch.settings.max_cholesky_size(100000000):
            eval(CG_model, likelihood, name="CG")
            plt.show()

        print('post training lengthscale = %.2f' % CG_model.covar_module.base_kernel.lengthscale.item())
        recovered_lengthscales_CG[i, c] = CG_model.covar_module.base_kernel.lengthscale.item()
        training_time_CG[i, c] = train_time

# %%

true_vec_RFF = np.repeat(lengthscale_vec, recovered_lengthscales_RFF.shape[1])
recovered_vec_RFF = recovered_lengthscales_RFF.reshape(
    recovered_lengthscales_RFF.shape[0] * recovered_lengthscales_RFF.shape[1])
RFF_samples_vec = np.tile(RFF_samples,
                          recovered_lengthscales_RFF.shape[0])
runtime_vec_RFF = training_time_RFF.reshape(
    training_time_RFF.shape[0] * training_time_RFF.shape[1])
print(recovered_vec_RFF.shape)

# %%

true_vec_CG = np.repeat(lengthscale_vec, recovered_lengthscales_CG.shape[1])
recovered_vec_CG = recovered_lengthscales_CG.reshape(
    recovered_lengthscales_CG.shape[0] * recovered_lengthscales_CG.shape[1])
runtime_vec_CG = training_time_CG.reshape(
    training_time_CG.shape[0] * training_time_CG.shape[1])
CG_iters_vec = np.tile(CG_iters,
                       recovered_lengthscales_CG.shape[0])
print(recovered_vec_CG.shape)

# %%

fig, ax = plt.subplots(1, 2, figsize=(7.5, 5))

# RFF
ax[0].plot([0, lengthscale_vec[-1]], [0, lengthscale_vec[-1]], 'gray')
scat0 = ax[0].scatter(true_vec_RFF, recovered_vec_RFF,
                      c=RFF_samples_vec, s=runtime_vec_RFF * 10,
                      cmap=plt.cm.get_cmap('coolwarm', len(RFF_samples)), alpha=.5)
fig.colorbar(scat0, ticks=RFF_samples, label='N RFF samples', ax=ax[0])
ax[0].set_xlabel("true")
ax[0].set_ylabel("learned")
ax[0].set_title("RFF in training")

# CG
ax[1].plot([0, lengthscale_vec[-1]], [0, lengthscale_vec[-1]], 'gray')
scat1 = ax[1].scatter(true_vec_CG, recovered_vec_CG,
                      c=CG_iters_vec, s=runtime_vec_CG * 10,
                      cmap=plt.cm.get_cmap('coolwarm', len(CG_iters)), alpha=.5)
fig.colorbar(scat1, ticks=CG_iters, label='N CG iters', ax=ax[1])
ax[1].set_xlabel("true")
ax[1].set_ylabel("learned")
ax[1].set_title("CG in training")
fig.suptitle('Recovering lengthscale (size ' + r'$\propto$' + ' runtime)')
fig.tight_layout()
plt.savefig('RFF_CG_initial_sim.png')

# %%

recovered_vec_RFF.shape
print(true_vec_RFF.shape)
print(RFF_samples.shape)
print(lengthscale_vec.shape)

# %%

# division plot
# print(recovered_lengthscales_RFF)
lengthscale_mat_RFF = np.tile(lengthscale_vec.reshape(-1, 1),
                              (1, RFF_samples.shape[0]))
div_error_RFF = recovered_lengthscales_RFF / lengthscale_mat_RFF
lengthscale_mat_CG = np.tile(lengthscale_vec.reshape(-1, 1),
                             (1, CG_iters.shape[0]))
div_error_CG = recovered_lengthscales_CG / lengthscale_mat_CG
# print(lengthscale_mat)

# %%

import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 3.0

# %%

fig, ax = plt.subplots(1, 2, figsize=(7.5, 5))
ax[0].axhline(y=0.0, linestyle='dashed', color='black');
ax[0].plot(RFF_samples, np.log(div_error_RFF.T), '--o', color='gray');
ax[0].plot(RFF_samples, np.log(np.mean(div_error_RFF, axis=0)), '-o', color='red')
ax[0].set_xticks(RFF_samples[np.arange(0, len(RFF_samples), 2)])
ax[0].set_xticklabels(RFF_samples[np.arange(0, len(RFF_samples), 2)], rotation=60)
ax[0].set_title('RFF')
ax[0].set_xlabel('N RFF samples', fontsize=14)
ax[0].set_ylabel(r'$\log(\hat{l^2}/{l^2})$', fontsize=16)
ax[1].axhline(y=0.0, linestyle='dashed', color='black');
ax[1].plot(CG_iters, np.log(div_error_CG.T), '--o', color='gray');
ax[1].plot(CG_iters, np.log(np.mean(div_error_CG, axis=0)), '-o', color="red")
ax[1].set_xticks(CG_iters);
ax[1].set_title('CG')
ax[1].set_xlabel('N CG iters', fontsize=14)
ax[1].set_ylabel(r'$\log(\hat{l^2}/{l^2})$', fontsize=16)
fig.suptitle('Recovering lengthscale')
fig.tight_layout()
plt.savefig('div_err_more_iters.png')

# %%


# %%

# division:
plt.plot(RFF_samples_vec,
         recovered_vec_RFF / true_vec_RFF)

# %%

plt.imshow(training_time_RFF)
plt.colorbar()
plt.ylabel('lengthscale')
plt.xlabel('RFF samples')
plt.title('Time')

# %%

plt.imshow(training_time_CG)
plt.colorbar()
plt.ylabel('lengthscale')
plt.xlabel('CG iters')
plt.title('Time')

# %%

train_x = torch.linspace(-5, 5, 200)
train_y = torch.sin(train_x * (5 * math.pi)) * train_x + torch.randn_like(train_x) * 0.1
test_x = torch.linspace(-6, 7, 151)

# %%

# re-initialize true model
true_likelihood = gpytorch.likelihoods.GaussianLikelihood()
true_model = True_GPRegressionModel(train_x, train_y, likelihood)
hypers = {
    'likelihood.noise_covar.noise': torch.tensor(.0001),
    'covar_module.base_kernel.lengthscale': torch.tensor(2.),
    'covar_module.outputscale': torch.tensor(20.),
}

true_model.initialize(**hypers)
print(
    true_model.likelihood.noise_covar.noise.item(),
    true_model.covar_module.base_kernel.lengthscale.item(),
    true_model.covar_module.outputscale.item()
)

true_model.eval()
true_likelihood.eval()

preds = true_likelihood(true_model(train_x))
train_y = preds.sample()  # true y, make sure that there are no duplicates

plt.plot(train_x, GP_sample.detach().numpy(), 'x')
plt.title('lengthscale = %.2f' % true_model.covar_module.base_kernel.lengthscale.item())
plt.show()

# %%


# %%


# %%


# %%


# %%

"""
