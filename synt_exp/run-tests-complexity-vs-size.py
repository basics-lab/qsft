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
from synt_src.synthetic_helper import SyntheticHelper
from qsft.parallel_tests import run_tests
from synt_src.synthetic_signal import generate_signal_w


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--num_subsample', type=int, nargs="+")
    parser.add_argument('--num_repeat', type=int, nargs="+")
    parser.add_argument('--b', type=int, nargs="+")
    parser.add_argument('--a', type=int)
    parser.add_argument('--snr', type=float)
    parser.add_argument('--n', type=int, nargs="+")
    parser.add_argument('--q', type=int)
    parser.add_argument('--t', type=int)
    parser.add_argument('--sparsity', type=int)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--subsampling', type=int, default=True)
    parser.add_argument('--jobid', type=int)

    args = parser.parse_args()
    debug = args.debug
    if debug:
        args.num_subsample = [2]
        args.num_repeat = [2]
        args.b = [6]
        args.a = 1
        args.n = np.linspace(40, 40, num=1, dtype=int)
        args.q = 3
        args.t = 5
        args.sparsity = 1
        args.snr = 50
        args.iters = 1
        args.jobid = "debug-" + str(uuid.uuid1())[:8]
        args.subsampling = True

    if debug:
        exp_dir_base = Path(f"results/{str(args.jobid)}")
    else:
        exp_dir_base = Path(f"/global/scratch/users/erginbas/qsft/synt-exp-results/{str(args.jobid)}")

    exp_dir_base.mkdir(parents=True, exist_ok=True)
    (exp_dir_base / "figs").mkdir(exist_ok=True)

    print("Parameters :", args, flush=True)

    methods = ["qsft_coded"]

    dataframes = []

    print("Starting the tests...", flush=True)

    for n_idx in range(len(args.n)):

        n = int(args.n[n_idx])

        noise_sd = np.sqrt((args.sparsity * args.a ** 2) / (10 ** (args.snr / 10)))

        print(fr"n = {n}, N = {args.q ** n:.2e}, sigma = {noise_sd}")

        b_valid = [b for b in args.b if b <= n]

        subsampling_args = {
            "num_subsample": max(args.num_subsample),
            "num_repeat": max(args.num_repeat),
            "b": max(b_valid),
            "all_bs": b_valid
        }

        test_args = {
            "n_samples": 200000
        }


        for it in range(args.iters):
            exp_dir = exp_dir_base / f"n{n}_i{it}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            _, locq, strengths = generate_signal_w(n, args.q, args.sparsity, args.a, args.a,
                                                   full=False, max_weight=args.t)

            signal_args = {
                "n": n,
                "q": args.q,
                "t": args.t,
                "locq": locq,
                "strengths": strengths,
            }

            helper = SyntheticHelper(signal_args=signal_args, methods=methods, subsampling=args.subsampling,
                                     exp_dir=exp_dir, subsampling_args=subsampling_args, test_args=test_args)

            for method in methods:
                if method == "lasso" and args.q ** n > 8000:
                    pass
                else:
                    dataframes.append(run_tests(method, helper, 1, args.num_subsample, args.num_repeat,
                                                b_valid, [noise_sd], parallel=False))


    results_df = pd.concat(dataframes, ignore_index=True)
    results_df.to_pickle(exp_dir_base / "result.pkl")