import fire

# Run Exact negative MLL v.s. number of optimization iterations
# Under different settings: RRCG (rr-iter-mean, rr-iter-std, lr), CG(num-cg), Cholesky


def main(dataset='pol', keops=True, toy_setup=False):
    hyper_trace_dir = f"./hyper_trace"

    if toy_setup:
        total_n = 100
        total_iters_range = [10]
        lr_range = [5e-2, 1e-2]
        rr_temp_range = [0.1]
        rr_iter_min_range = [10, 30]
        cg_iter_range = [20]
    else:
        total_n = -1
        total_iters_range = [1500]
        lr_range = [1e-2]
        rr_temp_range = [0.1, 0.05]
        #rr_temp_range = [0.1, 0.05]
        rr_iter_min_range = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        cg_iter_range = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    with open("run_multiple_rr_iters.sh", "w") as f:
        f.write("#!/bin/bash\n")

        f.write(f"\n\n##############################\n # Dataset {dataset} #\n##############################\n")

        # run rrcg_optimization
        for lr in lr_range:
            for total_iters in total_iters_range:

                # RRCG
                f.write("\n####### RRCG ######\n")
                for temp in rr_temp_range:
                    for rr_iter_min in rr_iter_min_range:

                        method = 'rrcg'
                        f.write(f"python run_rrcg_optimization.py --dataset={dataset} --total-n={total_n} --method={method} --rr_dist_type=expdecay --temp={temp} --rr-iter-min={rr_iter_min} --kernel-type=rbf --total-iters={total_iters} --lr={lr} --eval=False --save-hyper-trace=True --keops={keops}\n")

                # CG

                f.write("\n####### CG #######\n")
                for num_cg in cg_iter_range:
                    method = 'cg'
                    f.write(f"python run_rrcg_optimization.py --dataset={dataset} --total-n={total_n} --method={method} --num-cg={num_cg} --kernel-type=rbf --total-iters={total_iters} --lr={lr} --eval=False --save-hyper-trace=True --keops={keops}\n")

        # run get_exact_loss

        f.write("\nsleep 5s\n")

        f.write(f"python run_multiple_rr_iters.py --hyper-trace-dir={hyper_trace_dir} --keops={keops}\n")




if __name__ == "__main__":
    fire.Fire(main)