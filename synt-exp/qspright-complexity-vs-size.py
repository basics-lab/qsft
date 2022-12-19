#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import uuid

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',504)
pd.set_option('display.width',1000)

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
    args.num_subsample = [1, 2, 3, 4]
    args.num_random_delays = [1, 2, 3, 4]
    args.b = [4, 5, 6]
    args.a = 1      # synthetic signal size
    args.n = np.linspace(7, 15, num=5, dtype=int)
    args.q = 2
    args.sparsity = 10
    args.noise_sd = [1e-3]
    args.iters = 10
    args.jobid = "debug-" + str(uuid.uuid1())[:8]
    args.subsampling = True

if debug:
    exp_dir_base = Path(f"results/{str(args.jobid)}")
else:
    exp_dir_base = Path(f"/global/scratch/users/erginbas/qspright/synt-exp-results/{str(args.jobid)}")

exp_dir_base.mkdir(parents=True, exist_ok=True)
(exp_dir_base / "figs").mkdir(exist_ok=True)

print("Parameters :", args, flush=True)

method = "qspright"

min_samples = np.inf * np.ones(len(args.n))
min_times = np.inf * np.ones(len(args.n))

for n_idx in range(len(args.n)):

    n = args.n[n_idx]

    exp_dir = exp_dir_base / f"n{n}"
    query_args = {
        "query_method": "complex",
        "delays_method": "nso",
        "num_subsample": max(args.num_subsample),
        "num_random_delays": max(args.num_random_delays),
        "b": max(args.b),
        "all_bs": args.b
    }

    test_args = {
        "n_samples": 50000
    }

    print("Loading/Calculating data...", flush=True)

    exp_dir.mkdir(parents=True, exist_ok=True)

    print("n = {}, N = {:.2e}".format(n, args.q ** n))
    print("Starting the tests...", flush=True)

    dataframes = []

    for _ in range(args.iters):

        helper = SyntheticHelper(n, args.q, noise_sd=args.noise_sd[0], sparsity=args.sparsity,
                                 a_min=args.a, a_max=args.a,
                                 baseline_methods=[method], subsampling=args.subsampling,
                                 exp_dir=exp_dir, query_args=query_args, test_args=test_args)

        dataframes.append(run_tests(method, helper, 3, args.num_subsample, args.num_random_delays,
                                   args.b, args.noise_sd, parallel=False))

    results_df = pd.concat(dataframes, ignore_index=True)

    means = results_df.groupby(["num_subsample", "num_random_delay", "b", "noise_sd"], as_index=False).mean()
    stds = results_df.groupby(["num_subsample", "num_random_delay", "b", "noise_sd"], as_index=False).std()

    for i in means.index:
        mean_row = means.iloc[i]
        if mean_row['nmse'] < 1e-3:
            min_samples[n_idx] = min(min_samples[n_idx], mean_row['n_samples'])
            min_times[n_idx] = min(min_times[n_idx], mean_row['runtime'])

fig, ax = plt.subplots()

ax.plot(args.n, min_samples, "o-")
ax.set_xlabel('n')
ax.set_ylabel('Sample Complexity')
ax.set_yscale("log")
plt.grid()
plt.savefig(exp_dir_base / f'figs/sample-complexity-vs-n.png')
plt.show()

fig, ax = plt.subplots()

ax.plot(args.n, min_times, "o-")
ax.set_xlabel('n')
ax.set_ylabel('Run-Time Complexity')
plt.grid()
plt.savefig(exp_dir_base / f'figs/runtime-complexity-vs-n.png')
plt.show()

