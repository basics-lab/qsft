import numpy as np
import sys
import pandas as pd
import uuid

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 504)
pd.set_option('display.width', 1000)

sys.path.append("..")
sys.path.append("../src")

import argparse
from pathlib import Path
from rna_transform.rna_helper import RNAHelper
from qsft.parallel_tests import run_tests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--num_subsample', type=int, nargs="+")
    parser.add_argument('--num_repeat', type=int, nargs="+")
    parser.add_argument('--b', type=int, nargs="+")
    parser.add_argument('--noise_sd', type=float, nargs="+")
    parser.add_argument('--n', type=int, nargs="+")
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--subsampling', type=int, default=True)
    parser.add_argument('--jobid', type=int)

    args = parser.parse_args()
    debug = args.debug
    if debug:
        args.num_subsample = [4]
        args.num_repeat = [3]
        args.b = [7]
        args.n = [15]
        args.iters = 1
        args.jobid = "debug-" + str(uuid.uuid1())[:8]
        args.subsampling = True

    args.noise_sd = np.logspace(-3, -5, num=10)

    if debug:
        exp_dir_base = Path(f"results/{str(args.jobid)}")
    else:
        exp_dir_base = Path(f"/global/scratch/users/erginbas/qsft/synt-exp-results/{str(args.jobid)}")

    args.q = 4

    exp_dir_base.mkdir(parents=True, exist_ok=True)
    (exp_dir_base / "figs").mkdir(exist_ok=True)

    print("Parameters :", args, flush=True)

    methods = ["qsft"]

    dataframes = []

    print("Starting the tests...", flush=True)

    for n_idx in range(len(args.n)):

        n = args.n[n_idx]

        b_valid = [b for b in args.b if b <= n]

        subsampling_args = {
            "num_subsample": max(args.num_subsample),
            "num_repeat": max(args.num_repeat),
            "b": max(b_valid),
            "all_bs": b_valid
        }

        test_args = {
            "n_samples": 5000
        }

        print()
        print(fr"n = {n}, N = {args.q ** n}, sigma = {args.noise_sd}")

        for it in range(args.iters):
            exp_dir = exp_dir_base / f"i{it}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            signal_args = {
                "n": n,
                "q": args.q
            }

            helper = RNAHelper(signal_args=signal_args, methods=methods, subsampling=args.subsampling,
                               exp_dir=exp_dir, subsampling_args=subsampling_args, test_args=test_args)

            for method in methods:
                if method == "lasso" and args.q ** n > 3000:
                    pass
                else:
                    dataframes.append(run_tests(method, helper, 1, args.num_subsample, args.num_repeat,
                                                b_valid, args.noise_sd, parallel=False))

    results_df = pd.concat(dataframes, ignore_index=True)
    print()
    print(results_df)
    results_df.to_pickle(exp_dir_base / "result.pkl")