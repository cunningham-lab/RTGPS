import pickle
import time
import warnings
import gc
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import gpytorch as gpt
from gpytorch.distributions import base_distributions
from rff.rff_fns import OneOverJ
from experiments.rr_rff import RFFKernel
from experiments.rr_rff import RR_RFF_Kernel
from experiments.utils import print_time_taken


def get_test_rmse_nll_gp(train_ds, test_ds, settings, logger):
    likelihood = gpt.likelihoods.GaussianLikelihood()
    use_ard = settings["model_name"] == "rff_ard"
    model = GPRegressionModel(*train_ds, likelihood, use_ard=use_ard)
    set_hypers(model, settings["noise"], settings["ls"], settings["os"])
    likelihood = likelihood.cuda() if settings["cuda"] else likelihood
    model = model.cuda() if settings["cuda"] else model

    likelihood.eval()
    model.eval()
    t0 = time.time()
    rmse, nll = compute_test_metrics(model, test_ds)
    print_time_taken(t0, time.time(), logger=logger)
    return rmse, nll


def compute_test_metrics(model, test_ds):
    with torch.no_grad(), gpt.settings.fast_pred_var():
        with gpt.beta_features.checkpoint_kernel(0):
            y_hat = model(test_ds[0])
            nll = compute_nll(model, test_ds[1], y_hat)
            rmse = float(compute_rmse(y_hat.mean - test_ds[1]))
    return rmse, nll


def compute_nll(model, test_y, y_hat):
    mean = y_hat.mean
    covar = y_hat.lazy_covariance_matrix
    covar += model.likelihood.noise_covar(shape=mean.shape)
    y_hat.__class__(mean, covar)
    indep_dist = base_distributions.Normal(
        y_hat.mean, y_hat.variance.clamp_min(1e-8).sqrt()
    )
    nll = indep_dist.log_prob(test_y)
    nll = float(-torch.mean(nll))
    return nll


def compute_rmse(delta):
    rmse = torch.sqrt(torch.mean(torch.pow(delta, 2)))
    return rmse.item()


def fit_gp(train_ds, settings, logger, save_pickle=True):
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])
    use_cuda = settings["cuda"]
    likelihood = gpt.likelihoods.GaussianLikelihood()
    likelihood = likelihood.cuda() if use_cuda else likelihood

    model = select_model(settings, train_ds, likelihood)
    cholesky_model = GPRegressionModel(*train_ds, likelihood)
    cholesky_model = cholesky_model.cuda() if use_cuda else cholesky_model
    t0 = time.time()
    tr = TrainGPRR(model, settings, logger, cholesky_model)
    set_training_context(settings["model_name"], tr)

    print_time_taken(t0, time.time(), logger=logger)
    logger.info(get_hypers(model))
    if save_pickle:
        output_file = logger.log_file_name[:-4] + ".pkl"
        save_results(tr.results, output_file)


def set_training_context(model_name, tr):
    if model_name == "cholesky":
        with gpt.settings.max_cholesky_size(int(1.0e7)):
            tr.train()
    else:
        with gpt.settings.debug(False):
            tr.train()


def select_model(settings, train_ds, likelihood):
    model_name = settings["model_name"]
    use_cuda = settings["cuda"]
    if model_name == "cholesky":
        model = GPRegressionModel(*train_ds, likelihood)
    elif model_name == "rff":
        num_rff_samples = settings["rff_samples"]
        model = GPRegressionModelRFF(*train_ds, likelihood, num_rff_samples)
    elif model_name == "rff_ard":
        num_rff_samples = settings["rff_samples"]
        model = GPRegressionModelRFF(
            *train_ds, likelihood, num_rff_samples, use_ard=True
        )
    elif model_name == "ssrff":
        truncation_dist = get_truncation_dist(settings)
        model = GPRegressionModel_RR_RFF(
            *train_ds, likelihood, truncation_dist, single_sample=True, use_ard=False
        )
        if settings["warmup"]:
            model = warmup_model(model, settings["dataset_name"])

    elif model_name == "ssrff_ard":
        truncation_dist = get_truncation_dist(settings)
        model = GPRegressionModel_RR_RFF(
            *train_ds, likelihood, truncation_dist, single_sample=True, use_ard=True
        )
    model = model.cuda() if use_cuda else model
    return model


def get_truncation_dist(settings):
    dist_name = settings["truncation_name"]
    if dist_name == "onej":
        kwargs = settings["trunc_settings"]
        truncation_dist = OneOverJ(**kwargs)
    return truncation_dist


def warmup_model(model, dataset_name):
    input_location = "./results/warmup/" + dataset_name + ".pkl"
    results = load_object(input_location)
    set_hypers(model, results["noise"], results["ls"], results["output_scale"])
    return model


def load_object(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def run_rr_rounds(num_rr_rounds, model, exact_backup_model, train_ds, use_cuda):
    """computes RR estimators num_rr_rounds times, doing so for logdet and invquad"""
    res_dict = {}
    for key in ["J", "logdet", "invquad"]:
        res_dict[key] = np.zeros(num_rr_rounds)

    for i in range(num_rr_rounds):
        j, logdet, invquad = evaluate_RT_rff(model, exact_backup_model, train_ds)
        res_dict["J"][i] = j
        res_dict["logdet"][i] = logdet.cpu().item()
        res_dict["invquad"][i] = invquad.cpu().item()
        if use_cuda:
            # print(torch.cuda.memory_allocated())
            gc.collect()
            torch.cuda.empty_cache()

    return res_dict


def run_rff_rounds(num_rr_rounds, model, train_ds, use_cuda):
    """model is assumed to have the right number of samples"""
    res_dict = {}
    for key in ["J", "logdet", "invquad"]:
        res_dict[key] = np.zeros(num_rr_rounds)

    for i in range(num_rr_rounds):
        logdet, invquad = compute_logdet_invquad(model, train_ds)
        res_dict["logdet"][i] = logdet.cpu().item()
        res_dict["invquad"][i] = invquad.cpu().item()

        j = model.covar_module.base_kernel.num_samples
        res_dict["J"][i] = j

        if hasattr(model.covar_module.base_kernel, "randn_weights"):
            model.covar_module.base_kernel.randn_weights.normal_()

        if use_cuda:
            # print(torch.cuda.memory_allocated())
            gc.collect()
            torch.cuda.empty_cache()

    return res_dict


def evaluate_RT_rff(model, exact_backup_model, train_ds):
    """provides an rr estimate of logdet and invquad; also tracks Js.
    results should be compared to an RFF estimator with num_samples = torch.mean(J)"""
    log_det, inv_quad = compute_logdet_invquad(model, train_ds)
    J = model.covar_module.base_kernel.num_samples

    if J == model.dist_obj.max_val.item():
        log_det_exact, inv_quad_exact = compute_logdet_invquad(
            exact_backup_model, train_ds
        )
        assert log_det_exact.numel() == 1
        assert inv_quad_exact.numel() == 1
        log_det[-1] = log_det_exact
        inv_quad[-1] = inv_quad_exact

    debiased_log_det = model.RT_estimator(log_det).detach().cpu()
    debiased_inv_quad = model.RT_estimator(inv_quad).detach().cpu()

    return J, debiased_log_det, debiased_inv_quad


def compute_logdet_invquad(model, train_ds):
    """for rr-rff: log_det.numel() == num_samples, pre rr-weighted sum
    for rff and exact: log_det.numel() == 1"""
    with gpt.settings.debug(False):
        with torch.no_grad(), gpt.settings.lazily_evaluate_kernels(False):
            with gpt.settings.prior_mode(True):
                out = model.likelihood(model(train_ds[0]))
                log_det = out.lazy_covariance_matrix.logdet()
                inv_quad = out.lazy_covariance_matrix.inv_quad(train_ds[1])

    return log_det, inv_quad


class TrainGPRR:
    def __init__(self, model, settings, logger, cholesky_model=None):
        total_iters = settings["total_iters"]
        lr = settings["lr"]
        lr_wd = settings["lr_wd"]
        mil = settings["mil"]
        self.total_iters = settings["total_iters"]
        self.model = model
        self.logger = logger
        self.train_x, self.train_y = model.train_inputs[0], model.train_targets
        params = model.parameters()
        self.optimizer = select_optimizer(settings["optimizer"], params, lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=mil, gamma=lr_wd
        )
        self.mll = gpt.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        self.mll_chol = gpt.mlls.ExactMarginalLogLikelihood(
            cholesky_model.likelihood, cholesky_model
        )
        self.loss, self.ls, self.os, self.noise = 0.0, 0.0, 0.0, 0.0
        self.iter = 0
        num_dims = self.train_x.shape[1] if settings["model_name"] == "rff_ard" else 1
        self.results = {
            "ls": np.zeros((num_dims, total_iters)),
            "os": np.zeros(total_iters),
            "noise": np.zeros(total_iters),
            "loss": np.zeros(total_iters),
        }

    def train(self):
        self.model.train()
        for _ in range(self.total_iters):
            tic = time.time()
            self.take_one_step()
            toc = time.time()
            self.scheduler.step()
            self.update_trackers(time_taken=toc - tic)

    def take_one_step(self):
        self.optimizer.zero_grad()
        reset_rff_weights(self.model)
        y_hat = self.model(self.train_x)
        self.loss = -self.mll(y_hat, self.train_y)
        if hasattr(self.model, "dist_obj"):
            self.loss = add_weigths_to_loss(self.loss, self.model)
        self.loss.backward()
        self.optimizer.step()

    def update_trackers(self, time_taken):
        self.update_params()
        update_logger(
            self.logger,
            self.loss.item(),
            self.ls,
            self.os,
            self.noise,
            self.iter,
            time_taken,
        )
        self.update_results()

    def update_params(self):
        # outputscale = 0.62923026
        # self.model.covar_module.outputscale = outputscale
        self.ls = offload_ls(self.model.covar_module.base_kernel.lengthscale)
        self.os = self.model.covar_module.outputscale.item()
        self.noise = self.model.likelihood.noise.item()

    def update_results(self):
        self.results["ls"][:, self.iter] = self.ls
        self.results["os"][self.iter] = self.os
        self.results["noise"][self.iter] = self.noise
        self.results["loss"][self.iter] = self.loss
        self.iter += 1


def select_optimizer(name, params, lr):
    if name == "Adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif name == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr)
    return optimizer


def offload_ls(ls):
    ls = ls.detach().cpu().numpy()
    return ls


def update_logger(logger, loss, ls, os, noise, iteration, time_taken):
    text = f"iter: {iteration:4d} | loss: {loss:+1.4e} | "
    text += f"ls: {ls[0, 0]:4.6f} | "
    text += f"noise: {noise:4.6f} | os: {os:4.6f} | "
    text += f"time: {time_taken:4.2f} sec"
    logger.info(text)


def add_weigths_to_loss(loss, model):
    unbiased = model.RT_estimator(loss)
    return unbiased


def train(model, likelihood, name="", total_iters=500):
    train_x, train_y = model.train_inputs[0], model.train_targets
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpt.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    iterator = tqdm(range(total_iters), desc=f"{name} Training")
    output = model(train_x)
    loss = -mll(output, train_y)

    for _ in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        # if hasattr(model, "dist_obj"):
        #     loss = add_rr_weigths_to_loss(loss, model)
        loss.backward()
        optimizer.step()
        update_iterator_print(
            iterator,
            loss.item(),
            noise=likelihood.noise.item(),
            ls=model.covar_module.base_kernel.lengthscale.item(),
            os=model.covar_module.outputscale.item(),
        )
        reset_rff_weights(model)

    return mll


def update_iterator_print(iterator, loss, ls, os, noise):
    iterator.set_postfix(loss=loss, ls=ls, os=os, noise=noise)


def reset_rff_weights(model):
    if hasattr(model.covar_module.base_kernel, "randn_weights"):
        model.covar_module.base_kernel.randn_weights.normal_()


def get_hypers(model):
    hypers = {}
    hypers["noise_scale"] = model.likelihood.noise_covar.noise.cpu().item()
    hypers["output_scale"] = model.covar_module.outputscale.cpu().item()
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    is_using_ard = ls.shape[1] > 1
    if is_using_ard:
        for i in range(ls.shape[1]):
            key = "ls_" + str(i)
            hypers[key] = ls[0, i]
    else:
        hypers["ls"] = ls.item()
    return hypers


def build_RFF_model(hypers, num_rff_samples, train_ds, use_cuda):
    likelihood = gpt.likelihoods.GaussianLikelihood()
    likelihood = likelihood.cuda() if use_cuda else likelihood
    RFF_model = GPRegressionModelRFF(*train_ds, likelihood, num_rff_samples)
    RFF_model = RFF_model.cuda() if use_cuda else RFF_model
    set_hypers(RFF_model, hypers["noise_scale"], hypers["ls"], hypers["output_scale"])
    return RFF_model


def build_RR_RFF_model(train_ds, dist_obj, single_sample, hypers, use_cuda):
    likelihood = gpt.likelihoods.GaussianLikelihood()
    likelihood = likelihood.cuda() if use_cuda else likelihood
    model = GPRegressionModel_RR_RFF(
        train_ds[0], train_ds[1], likelihood, dist_obj, single_sample
    )
    model = model.cuda() if use_cuda else model
    if hypers is not None:
        set_hypers(model, hypers["noise_scale"], hypers["ls"], hypers["output_scale"])
    return model


class GPRegressionModel_RR_RFF(gpt.models.ExactGP):
    def __init__(
            self, train_x, train_y, likelihood, dist_obj, single_sample, use_ard=False
    ):
        super().__init__(train_x, train_y, likelihood)
        self.dist_obj = dist_obj
        self.mean_module = gpt.means.ZeroMean()
        if use_ard:
            rr_rff_kernel = RR_RFF_Kernel(
                single_sample=single_sample,
                min_val=self.dist_obj.min_val,
                ard_num_dims=train_x.shape[1],
            )
        else:
            rr_rff_kernel = RR_RFF_Kernel(
                single_sample=single_sample, min_val=self.dist_obj.min_val
            )
        self.covar_module = gpt.kernels.ScaleKernel(rr_rff_kernel)

    def single_sample_estimator(self, terms):
        base_term = terms[0]
        final_delta = terms[-1] - terms[-2]
        ss_weight = 1.0 / self.dist_obj.prob(self.dist_obj.index_sampled)
        ss_estimator = base_term + ss_weight * final_delta
        return ss_estimator

    def RT_estimator(self, input_terms):
        ss_estimator = self.single_sample_estimator(input_terms)
        return ss_estimator

    def forward(self, x):
        mean_x = self.mean_module(x)
        self.dist_obj.sample()
        J = self.dist_obj.value_sampled
        self.covar_module.base_kernel.num_samples = int(J.item())
        with gpt.settings.lazily_evaluate_kernels(False):
            covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)


def randomly_sample_hypers(total_rounds, case="all", std=0.1):
    ls = torch.exp(torch.normal(mean=0.0, std=std, size=(total_rounds,)))
    if case == "all":
        noise_scale = torch.exp(torch.normal(mean=0.0, std=std, size=(total_rounds,)))
        output_scale = torch.exp(torch.normal(mean=0.0, std=std, size=(total_rounds,)))
    else:
        noise_scale = 0.01 * torch.ones_like(ls)
        output_scale = torch.ones_like(ls)
    return noise_scale, ls, output_scale


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_ax(v, div, axs, desc, ylims):
    axs.axhline(y=0.0, linestyle="dashed", color="black")
    axs.set_title(desc["title"])
    axs.set_xlabel(desc["xlabel"], fontsize=14)
    axs.set_ylabel(desc["ylabel"], fontsize=16)
    # axs.plot(v, div.T, '--o', color='gray')
    # axs.plot(v, np.median(div, axis=0), '-o', color='red')
    median, std = np.median(div, axis=0), np.std(div, axis=0)
    axs.vlines(v, ymin=median - std, ymax=median + std, color="gray")
    axs.plot(v, median, "-o", color="red")
    axs.set_ylim(ylims)
    axs.set_xticks(v[np.arange(0, len(v), 2)])
    axs.set_xticklabels(v[np.arange(0, len(v), 2)], rotation=60)
    return axs


def get_lower_upper_via_quantiles(v, q_min=0.20, q_max=0.80):
    v = v[np.logical_not(np.isnan(v))]
    v = v[np.logical_not(np.isinf(v))]
    lower = np.min(np.quantile(v, q=q_min, axis=0))
    lower = min(-0.01, lower)
    upper = np.max(np.quantile(v, q=q_max, axis=0))
    upper = max(0.01, upper)
    return lower, upper


class Tracker:
    def __init__(self, total_rounds, rff_samples, cg_iters):
        num_rff_samples = rff_samples.shape[0]
        num_cg_iters = cg_iters.shape[0]
        self.results = {
            "chol": {
                "inv_quad": np.zeros(total_rounds),
                "logdet": np.zeros(total_rounds),
            },
            "rff": {
                "inv_quad": np.zeros((total_rounds, num_rff_samples)),
                "logdet": np.zeros((total_rounds, num_rff_samples)),
            },
            "rff_samples": rff_samples,
            "cg": {
                "inv_quad": np.zeros((total_rounds, num_cg_iters)),
                "logdet": np.zeros((total_rounds, num_cg_iters)),
            },
            "cg_iters": cg_iters,
        }

    def save_results(self, output_file):
        with open(file=output_file, mode="wb") as f:
            pickle.dump(obj=self.results, file=f)


def recover_cholesky(hyperparams, train_ds, total_iters=500):
    use_cuda = torch.cuda.is_available()
    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(*train_ds, likelihood)
    set_hypers(model, *hyperparams)
    likelihood = likelihood.cuda() if use_cuda else likelihood
    model = model.cuda() if use_cuda else model
    print_initial_hypers(model)

    with gpt.settings.max_cholesky_size(int(1.0e7)):
        inv_quad, logdet = train(
            model, likelihood, name="Cholesky", total_iters=total_iters
        )
    ls = model.covar_module.base_kernel.lengthscale.item()
    print(f"Post training lengthscale = {ls:2.2f}")
    return model, ls, inv_quad, logdet


def recover_rff(num_rff_samples, hyperparams, train_ds, total_iters=500):
    use_cuda = torch.cuda.is_available()
    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = GPRegressionModelRFF(*train_ds, likelihood, num_rff_samples)
    likelihood = likelihood.cuda() if use_cuda else likelihood
    model = model.cuda() if use_cuda else model
    set_hypers(model, *hyperparams)
    print_initial_hypers(model)

    with gpt.settings.max_cholesky_size(int(1.0e7)):
        loss = train(model, likelihood, name="RFFs", total_iters=total_iters)
    ls = model.covar_module.base_kernel.lengthscale.item()
    print(f"Post training lengthscale = {ls:2.2f}")
    return model, ls, loss


def recover_cg(cg_iters, hyperparams, train_ds, total_iters=500):
    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(*train_ds, likelihood)
    set_hypers(model, *hyperparams)
    print_initial_hypers(model)

    # inv_quad, logdet = train_cg(model, likelihood, cg_iters, total_iters)
    loss = train_cg(model, likelihood, cg_iters, total_iters)
    ls = model.covar_module.base_kernel.lengthscale.item()
    print(f"Post training lengthscale = {ls:2.2f}")
    # return model, ls, inv_quad, logdet
    return model, ls, loss


def print_initial_hypers(model, print_ls_flag=True):
    noise_scale = model.likelihood.noise_covar.noise.item()
    output_scale = model.covar_module.outputscale.item()

    if print_ls_flag:
        ls = model.covar_module.base_kernel.lengthscale.item()
        print(
            f"Pre training: noise scale {noise_scale: 4.4f} | "
            + f"lengthscale {ls:4.4f} | "
            + f"output scale {output_scale:4.4f}"
        )
    else:
        print(
            f"Pre training: noise scale {noise_scale: 4.4f} | "
            + f"output scale {output_scale:4.4f}"
        )


def set_hypers(model, noise_scale, ls, output_scale):
    hypers = {
        "likelihood.noise_covar.noise": noise_scale,
        "covar_module.base_kernel.lengthscale": ls,
        "covar_module.outputscale": output_scale,
    }
    model.initialize(**hypers)


def plot_results(model, likelihood, test_x):
    train_x, train_y = model.train_inputs[0], model.train_targets
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    model.eval()

    with torch.no_grad(), gpt.settings.fast_pred_var():
        prediction = likelihood(model(test_x))
        mean = prediction.mean
        lower, upper = prediction.confidence_region()
        ax.plot(train_x.numpy(), train_y.numpy(), "k*", label="Training Data")
        ax.plot(test_x.numpy(), mean.numpy(), "b", label="Prediction")
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.legend(["Observed Data", "Mean", "Confidence"])
        ax.set(xlabel="x", ylabel="y", ylim=(-6.0, 6.0))

    return fig


def train_cg(model, likelihood, num_cg, total_iters):
    with gpt.settings.max_cholesky_size(0):
        warnings.simplefilter("ignore", gpt.utils.warnings.NumericalWarning)
        with gpt.settings.max_cg_iterations(num_cg):
            with gpt.settings.max_lanczos_quadrature_iterations(num_cg):
                with gpt.settings.cg_tolerance(1e-50):
                    with gpt.settings.max_preconditioner_size(0):
                        # inv_quad, logdet = train(model, likelihood, name="CG",
                        #                          total_iters=total_iters)
                        loss = train(
                            model, likelihood, name="CG", total_iters=total_iters
                        )
    # return inv_quad, logdet
    return loss


class GPRegressionModel(gpt.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, use_ard=False):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpt.means.ZeroMean()
        if use_ard:
            self.covar_module = gpt.kernels.ScaleKernel(
                gpt.kernels.RBFKernel(ard_num_dims=train_x.shape[1]),
            )
        else:
            self.covar_module = gpt.kernels.ScaleKernel(
                gpt.kernels.RBFKernel(),
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)


class GPRegressionModelRFF(GPRegressionModel):
    def __init__(self, train_x, train_y, likelihood, num_rff_samples, use_ard=False):
        super().__init__(train_x, train_y, likelihood)
        if use_ard:
            self.covar_module = gpt.kernels.ScaleKernel(
                RFFKernel(
                    num_samples=num_rff_samples, ard_num_dims=train_x.shape[1]
                )
            )
        else:
            self.covar_module = gpt.kernels.ScaleKernel(
                RFFKernel(num_samples=num_rff_samples)
            )


def compute_recovered_vs_true_matrix(recovered, true):
    ls_mat_rff = np.tile(true.reshape(-1, 1), (1, recovered.shape[1]))
    div_matrix = recovered / ls_mat_rff
    return div_matrix


def get_rff_samples(case):
    if case == "empty":
        rff_samples = np.arange(start=1, stop=1)
    elif case == "1":
        rff_samples = np.arange(start=1, stop=1, step=1)
    elif case == "2":
        rff_samples = np.arange(start=100, stop=1600, step=100)
    elif case == "3":
        rff_samples = np.arange(start=1, stop=100 + 1, step=5)
    elif case == "4":
        rff_samples = np.concatenate(
            (np.array([10, 50]), np.arange(100, 1000, 100), np.arange(1000, 3000, 250))
        )
    else:
        raise NotImplementedError
    return rff_samples


def get_cg_iters(case):
    if case == "empty":
        cg_iters = np.arange(start=1, stop=1)
    elif case == "1":
        cg_iters = np.arange(start=10, stop=20, step=2)
    elif case == "2":
        cg_iters = np.arange(start=5, stop=20, step=1)
    elif case == "3":
        cg_iters = np.arange(start=1, stop=20, step=1)
    else:
        raise NotImplementedError
    return cg_iters


def sample_from_prior(model, likelihood, train_x):
    with gpt.settings.prior_mode(True):
        prior_preds = likelihood(model(train_x))
    return prior_preds.sample()


def get_string_time_taken(tic, toc, experiment_name="Experiment"):
    minutes = round((toc - tic) / 60)
    seconds = (toc - tic) - minutes * 60
    return f'{experiment_name} took: {minutes:4d} min and {seconds:4.2f} sec'


def load_results(input_file):
    with open(input_file, mode="rb") as f:
        results = pickle.load(f)
    return results


def save_results(results, output_file):
    with open(file=output_file, mode="wb") as f:
        pickle.dump(obj=results, file=f)
