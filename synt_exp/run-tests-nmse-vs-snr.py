import numpy as np
import sys
import pandas as pd
import uuid
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 504)
pd.set_option('display.width', 1000)
sys.path.append("..")
import argparse
from pathlib import Path
from synt_exp.synt_src.synthetic_helper import SyntheticHelper
from qsft.parallel_tests import run_tests
from synt_exp.synt_src.synthetic_signal import generate_signal_w

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--num_subsample', type=int, nargs="+")
    parser.add_argument('--num_repeat', type=int, nargs="+")
    parser.add_argument('--b', type=int, nargs="+")
    parser.add_argument('--a', type=int)
    parser.add_argument('--snr', type=float, nargs="+")
    parser.add_argument('--n', type=int)
    parser.add_argument('--q', type=int)
    parser.add_argument('--sparsity', type=int, nargs="+")
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--subsampling', type=int, default=True)
    parser.add_argument('--jobid', type=int)

    args = parser.parse_args()
    debug = args.debug
    if debug:
        args.num_subsample = [3]
        args.num_repeat = [1]
        args.b = [7]
        args.a = 1
        args.n = 20
        args.q = 4
        args.sparsity = [100, 250, 1000]
        args.iters = 5
        args.jobid = "debug-" + str(uuid.uuid1())[:8]
        args.subsampling = True

    args.snr = np.linspace(-25, 5, num=30)

    if debug:
        exp_dir_base = Path(f"results/{str(args.jobid)}")
    else:
        exp_dir_base = Path(f"/global/scratch/users/erginbas/qsft/synt-exp-results/{str(args.jobid)}")

    exp_dir_base.mkdir(parents=True, exist_ok=True)
    (exp_dir_base / "figs").mkdir(exist_ok=True)

    print("Parameters :", args, flush=True)

    methods = ["qsft"]

    dataframes = []

    print("Starting the tests...", flush=True)

    subsampling_args = {
        "num_subsample": max(args.num_subsample),
        "num_repeat": max(args.num_repeat),
        "b": max(args.b),
        "all_bs": args.b
    }

    test_args = {
        "n_samples": 50000
    }

    print()
    print("n = {}, N = {:.2e}".format(args.n, args.q ** args.n))

    for s in range(len(args.sparsity)):

        sparsity = args.sparsity[s]
        noise_sd = np.sqrt((sparsity * args.a ** 2) / (args.q ** args.n * 10 ** (np.array(args.snr) / 10)))

        for it in range(args.iters):
            exp_dir = exp_dir_base / f"s{sparsity}_i{it}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            _, locq, strengths = generate_signal_w(args.n, args.q, sparsity, args.a, args.a, full=False)

            signal_args = {
                "n": args.n,
                "q": args.q,
                "locq": locq,
                "strengths": strengths,
            }

            helper = SyntheticHelper(signal_args=signal_args, methods=methods, subsampling=args.subsampling,
                                     exp_dir=exp_dir, subsampling_args=subsampling_args, test_args=test_args)

            for method in methods:
                run_df = run_tests(method, helper, 1, args.num_subsample, args.num_repeat,
                                        args.b, noise_sd, parallel=False)
                run_df["sparsity"] = sparsity
                dataframes.append(run_df)


    results_df = pd.concat(dataframes, ignore_index=True)
    results_df['snr'] = 10 * np.log10((results_df['sparsity'] / (results_df["q"] ** results_df["n"])) * \
                                        (args.a ** 2 / results_df["noise_sd"] ** 2))

    print(results_df)

    results_df.to_pickle(exp_dir_base / "result.pkl")
