# Bias-Free Scalable Gaussian Processes via Randomized Truncations
This repo contains the GPyTorch implementation of the RR-CG and SS-RFF

<br>**Abstract**<br>
*Scalable Gaussian Process methods are computationally attractive, yet introduce modeling
biases that require rigorous study.  This paper analyzes two common techniques: early
truncated conjugate gradients (CG) and random Fourier features (RFF).  We find that both
methods introduce a systematic bias on the learned hyperparameters: CG tends to underfit
while RFF tends to overfit.  We address these issues using randomized truncation
estimators that eliminate bias in exchange for increased variance.  In the case of RFF,
we show that the bias-to-variance conversion is indeed a trade-off: the additional
variance proves detrimental to optimization. However, in the case of CG, our unbiased
learning procedure meaningfully outperforms its biased counterpart with minimal
additional computation.*

## Requirements
Create a new conda environment with python>=3.6 as follows:

`conda create -n myenv python=3.6`

To activate the conda environment, run:
`conda activate myenv`

Ensure that you add the repo and the GPyTorch folders to the
`$PYTHONPATH` variable. You can do so by first running

`PYTHONPATH="path/to/repo/gpytorch"` 

then appending 

`PYTHONPATH=$PYTHONPATH:"path/to/repo/RR-GP"`.

For the paths to be recognised by python, please verify that you use your absolute paths from root directory.

To install the package requirements run the following
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
conda install faiss-gpu -c pytorch
bash create_dirs.sh
```
Note that installing `faiss-gpu` requires access to a GPU.

Here are the steps to download the UCI regression datasets for the experiments
 - the full set of twelve datasets can be downloaded at
   https://d2hg8soec8ck9v.cloudfront.net/datasets/uci_data.tar.gz
 - download the tar file into the `./datasets/` folder, and then run the following command from the `./datasets/` folder

```sh
tar -xvf uci_data.tar.gz && for dset in 3droad bike buzz elevators houseelectric keggdirected keggundirected kin40k pol protein slice song; do mv ${dset}/${dset}.mat ${dset}.mat && rm -rf ${dset}; done
```
Next, please copy all the matlab files in the `./datasets` folder to `./experiments/datasets`. This is a temporary inconvenience that will be removed in a future release.

## Experiments

### Structure of the repository
The four folders in needed to reproduce our experiments are \*.
* `baselines`: contains all the scripts to run the SVGP and SGGP baselines for section 5.3.
* `experiments`: contains all the scripts to run the RFF and SS-RFF  experiments as well
  as the scripts to generate the plots in the manuscript.
* `rrcg`: contains all the auxiliary functions to run the RR-CG experiments such at the
  truncation distribution class and the GPyTorch implementation of RR-CG.
* `rrcg_experiments`: contains all the scripts to run the RR-CG experiments.

\*If not mentioned, then these folders can be ignored as they are not related to the
experiments or are legacy code.

### Experiment scripts

All experiments should be run from the main repo folder, temporarily called RR-GP.

#### Running SVGP & SGGP

To run SGGP

`python baselines/run_sggp.py --dataset=pol --total_n=-1 --kernel_type=rbf`

To run SVGP

`python baselines/run_svgp.py --dataset=pol --total_n=-1 --kernel_type=rbf`

#### Running CG & RR-CG
All the basic functionalities for implementing CG and RR-CG including the truncation distribution are in the `./rrcg` folder. All the content to run experiments relating to CG / RR-CG / Cholesky is in the `./rrcg_experiments` folder. Specifically, the key file to run GP optimization is `./rrcg_experiments/run_rrcg_optimization.py`. 

To run GP optimization with CG,  we need to specify `--method=cg`, and number of CG iterations `--num-cg`. For example, 

```
python rrcg_experiments/run_rrcg_optimization.py --dataset=pol --total-iters=500 --lr=0.05 --method=cg --num-cg=100 --total_n=-1
```

To run GP optimization with RR-CG, we need to specify `--method=rrcg`, and the truncation distribution type `--rr-dist-type` along with the distribution parameters `**dist-of-iter-kwargs` (to view a list of truncation distributions and their parameters, please see `./rrcg/dist_of_iterations_for_rrcg.py`).  For example, 

```
python rrcg_experiments/run_rrcg_optimization.py --dataset=pol --total-iters=500 --lr=0.05 --method=rrcg --rr-dist-type=expdecay --temp=0.05  --rr-iter-min=80 --total_n=-1
```

To run GP optimization with Cholesky, we do not need to specify extra method-related parameters, and it is just as simple as 

```
python rrcg_experiments/run_rrcg_optimization.py --dataset=pol --total-iters=500 --lr=0.05 --method=cholesky --total_n=-1
```

Note that the `--total_n` argument allows using a subset of the dataset for faster testing.

#### Running RFF & SS-RFF

All the content for running RFF & SS-RFF is in the `./experiments` folder. To run a specific instance of the models, run

```
python experiments/run_rff.py
```
and you will have to change the configurations then you could modify the following
variables inside the script

| Name | Value (example) and type | Description |
| :-------------------------------- | :------------------------------: | :-----------------------: |
| `datasets` | `<dict> ('{1: 'pol'}')`  | Contains all the names of the datasets. |
| `dataset_name` | `<str> ('pol')`  | Dataset to run in the experiment. |
| `seeds` | `<list> ([5328, 5945])`  | List of seeds to run in the experiment. The length of the list is also the number of times that the experiment will be run. |
| `run_sample` | `<bool> (True)`  | Flag to run only a sample. Set to False to run all the data. |
| `sample_size` | `<int> (100)`  | Sample size (in case of running a sample) |
| `model_name` | `<str> ('rff_ard')`  | Model name. Options: `rff`, `rff_ard`, `ssrff`. |
| `rff_samples` | `<int> (200)`  | # of RFF features to use. |
| `total_iters` | `<int> (1000)`  | # of gradient descent iterations.  |
| `lr` | `<float> (0.01)`  | Learning rate value. |
| `lr_wd` | `<float> (0.01)`  | Learning rate decay that the scheduler uses. |
| `optimizer` | `<str> ('Adam')`  | Name of the optimizer to use. |
| `mil` | `<list> ([500, 700, 900])`  | Iterations where the scheduler decays the learning rate. |
| `truncation_name` | `<str> ('onej')`  | Name of the truncation distribution to use. |
| `trunc_settings` | `<dict> ({'min_val': 500, ...})`  | Specifications of the truncation dist. This includes the minimum value of features to use, the max value (most likely determined by the GPU memory - higher is better), the decay coefficient 1/J^{coeff} and the step size|

For running the experiments in the paper try
```
bash rff.sh
```

## Files
* `experiments/add_results.py`: helper script to fill in results dictionary
    * Extends the file `all_results.pkl`
* `experiments/capture_test_metrics.py`: creates database to print out results table
    * Creates `test.pkl`
* `experiments/compute_exact_loss.py`: helper script to add exact loss to results dictionary
    * This one takes a lot of time since it is running the full Cholesky at each iteration  for each model
* `experiments/construct_results.py`: helper script to add cholesky info to the results dictionary
    * For Figure 4 and 5. Outputs the `all_results.pkl`
* `experiments/create_loss_landscape.py`: generates the loss landscape for the specified hyperparameters
* `experiments/lengthscale_recovery.py`: runs the lengthscale recovery experiment
* `experiments/plot_fns.py`: utilities for plotting
* `experiments/test_models.py`: outputs the NLL and RMSE for a given model
* `experiments/utils`: general utils for the report tables
* `experiments/rff_tables.py`: table output for text print
    * Reads in `rffard_results.pkl` 
* `experiments/results_to_db.py`: pases results into CSV for LaTeX

First save the output in experiments construct results and then use compute exact loss.
This will create a dictionary with keys as the models and then each model will have a
dictionary with the loss, os, ls, noise, and exact loss as keys (per iteration)
