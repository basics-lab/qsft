#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import uuid

sys.path.append("..")
sys.path.append("../src")

from qspright.utils import best_convex_underestimator
import argparse
from pathlib import Path
from qspright.synthetic_helper import SyntheticHelper
from qspright.parallel_tests import run_tests

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--num_subsample', type=int, nargs="+")
parser.add_argument('--num_random_delays', type=int, nargs="+")
parser.add_argument('--b', type=int, nargs="+")
parser.add_argument('--a', type=int)
parser.add_argument('--noise_sd', type=float, nargs="+")
parser.add_argument('--n', type=int)
parser.add_argument('--q', type=int)
parser.add_argument('--sparsity', type=int)
parser.add_argument('--iters', type=int, default=1)
parser.add_argument('--subsampling', type=int, default=True)
parser.add_argument('--jobid', type=int)

args = parser.parse_args()
debug = args.debug
if debug:
    args.num_subsample = [2]
    args.num_random_delays = [2]
    args.b = [7]
    args.a = 1
    args.n = 10
    args.q = 4
    args.sparsity = 100
    args.iters = 1
    args.jobid = "debug-" + str(uuid.uuid1())[:8]
    args.subsampling = True

args.noise_sd = np.logspace(-5, 2, num=30)

if debug:
    exp_dir = Path(f"results/{str(args.jobid)}")
else:
    exp_dir = Path(f"/global/scratch/users/erginbas/qspright/synt-exp-results/{str(args.jobid)}")

print("Parameters :", args, flush=True)

query_args = {
    "query_method": "complex",
    "delays_method": "nso",
    "num_subsample": max(args.num_subsample),
    "num_random_delays": max(args.num_random_delays),
    "b": max(args.b),
    "all_bs": args.b
}

methods = ["qspright", "binary_qspright"]
colors = ["red", "blue"]

test_args = {
    "n_samples": 50000
}

print("Loading/Calculating data...", flush=True)

exp_dir.mkdir(parents=True, exist_ok=True)
(exp_dir / "figs").mkdir(exist_ok=True)

helper = SyntheticHelper(args.n, args.q, noise_sd=args.noise_sd[0], sparsity=args.sparsity,
                         a_min=args.a, a_max=args.a,
                         baseline_methods=methods, subsampling=args.subsampling,
                         exp_dir=exp_dir, query_args=query_args, test_args=test_args)

print("n = {}, N = {:.2e}".format(args.n, args.q ** args.n))

print("Starting the tests...", flush=True)

fig, ax = plt.subplots()

for m in range(len(methods)):
    # Test QSPRIGHT with different parameters
    # Construct a grid of parameters. For each entry, run multiple test rounds.
    # Compute the average for each parameter selection.
    results_df = run_tests(methods[m], helper, args.iters, args.num_subsample, args.num_random_delays,
                           args.b, args.noise_sd, parallel=False)

    # results_df.to_csv(f'results/{str(args.jobid)}/results_df_{methods[m]}.csv')

    means = results_df.groupby(["num_subsample", "num_random_delay", "b", "noise_sd"], as_index=False).mean()
    stds = results_df.groupby(["num_subsample", "num_random_delay", "b", "noise_sd"], as_index=False).std()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(results_df)

    x_values = []
    y_values = []

    for i in means.index:
        mean_row = means.iloc[i]
        std_row = stds.iloc[i]
        x_values.append(10 * np.log(args.a**2 / (2 * mean_row['noise_sd']**2)))
        y_values.append(mean_row['nmse'])

    ax.plot(x_values, y_values, "o-", color=colors[m], label=methods[m])

ax.set_xlabel('SNR')
ax.set_ylabel('Test NMSE')
plt.legend()
plt.grid()
plt.savefig(exp_dir / f'figs/nmse-vs-snr.png')
plt.show()

