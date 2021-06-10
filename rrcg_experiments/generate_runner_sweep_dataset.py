datasets_and_N_dict = {
    'pol': 9600,
    'elevators': 10623,
    'bike': 11122,

    'kin40k': 25600,
    'protein': 29267,
    'keggdirected': 31249,
    'slice': 34240,
    'keggundirected': 40709,

    '3droad': 278319,
    # 'song': 329820,
    'buzz': 373280,
    #'houseelectric': 1311539,
}

cholesky_available_datasets = ['pol', 'elevators', 'bike']


"""
def test_mean_and_std():
    for train_n in datasets_and_N_dict.values():
        for rr_iter_min in [10, 30, 50, 70, 90]:
            dist = ExpDecayDist(temp=0.1, min=rr_iter_min, max=train_n)
            print(f"train_n={train_n}, mean={round(dist.mean.item())}, std={round(dist.std.item())}")
        print("\n")
        
    for train_n in datasets_and_N_dict.values():
        for rr_iter_min in [1, 20, 40, 60, 80]:
            dist = ExpDecayDist(temp=0.05, min=rr_iter_min, max=train_n)
            print(f"train_n={train_n}, mean={round(dist.mean.item())}, std={round(dist.std.item())}")
        print("\n")
"""


dataset_to_run = ['pol', 'elevators', 'bike']
keops = 'True'
kernel_type = 'rbf'
save_model = 'True'
lr = 5e-2


seed_list = [10, 20, 30]

run_cholesky = False
num_cg_list = [] #[20, 40, 60, 80, 100]
mean_rrcg_list = [20] #[20, 40, 60, 100]
temp_rrcg_list = [0.05] #[0.1, 0.05]  # once mean and temp are specified, rr_min and std are determined


assert kernel_type in ['rbf', 'rbf-ard']
assert save_model in ['True', 'False']

with open("run_sweep_dataset.sh", "w") as f:
    f.write("#!/bin/bash\n")
    #f.write("exec &> logfile.txt\n")
    for dataset in dataset_to_run:
        train_n = datasets_and_N_dict[dataset]

        f.write(f"\n\n##############################\n # Dataset {dataset} #\n##############################\n")

        if train_n <= 2e4:
            if lr == 0.05:
                total_iters = 500
            elif lr == 0.01:
                total_iters = 1500
            else:
                raise ValueError("Please specified total_iters for this lr.")
        elif train_n <= 5e4:
            if lr == 0.05:
                total_iters = 200
            elif lr == 0.01:
                total_iters = 800
            else:
                raise ValueError("Please specified total_iters for this lr.")
        elif train_n <= 5e5:
            if lr == 0.05:
                total_iters = 100
            elif lr == 0.01:
                total_iters = 300
            else:
                raise ValueError("Please specified total_iters for this lr.")

        for seed in seed_list:

            # cholesky
            f.write("\n# Cholesky\n")
            if run_cholesky:
                if dataset in cholesky_available_datasets:
                    f.write(
                        f"python run_rrcg_optimization.py --dataset={dataset} --seed={seed} --total-n=-1 --method=cholesky --total-iters={total_iters} --lr={lr} --kernel-type={kernel_type} --keops={keops} --save-model={save_model}\n")

            # cg
            f.write("\n# CG\n")
            for num_cg in num_cg_list:
                f.write(
                    f"python run_rrcg_optimization.py --dataset={dataset} --seed={seed} --total-n=-1 --method=cg --num-cg={num_cg} --total-iters={total_iters} --lr={lr} --kernel-type={kernel_type} --keops={keops} --save-model={save_model}\n")

            # rrcg
            # Empirically, we found that
            ## temp = 0.1 -> std = 10, mean = min + 10
            ## temp = 0.05 -> std = 20, mean = min + 20
            f.write("\n# RR-CG\n")
            for temp in temp_rrcg_list:
                if temp == 0.1:
                    f.write(f"# temp = {temp} \n")
                    for mean_rr in mean_rrcg_list:
                        rr_iter_min = mean_rr - 10
                        f.write(
                            f"python run_rrcg_optimization.py --dataset={dataset} --seed={seed} --total_n=-1 --method=rrcg --total-iters={total_iters} --lr={lr} --rr_dist_type=expdecay --temp={temp} --rr-iter-min={rr_iter_min} --kernel-type={kernel_type} --keops={keops} --save-model={save_model}\n")
                else:
                    assert temp == 0.05
                    f.write(f"# temp = {temp} \n")
                    for mean_rr in mean_rrcg_list:
                        if mean_rr == 20:
                            rr_iter_min = 1
                        else:
                            rr_iter_min = mean_rr - 20
                        f.write(
                            f"python run_rrcg_optimization.py --dataset={dataset} --seed={seed} --total_n=-1 --method=rrcg --total-iters={total_iters} --lr={lr} --rr_dist_type=expdecay --temp={temp} --rr-iter-min={rr_iter_min} --kernel-type={kernel_type} --keops={keops} --save-model={save_model}\n")


