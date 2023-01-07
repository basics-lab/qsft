#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import uuid

sys.path.append("..")
sys.path.append("../src")

from rna_transform.rna_helper import RNAHelper
from qspright.utils import best_convex_underestimator
from qspright.parallel_tests import run_tests
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--num_subsample', type=int, nargs="+")
parser.add_argument('--num_repeat', type=int, nargs="+")
parser.add_argument('--b', type=int, nargs="+")
parser.add_argument('--noise_sd', type=float, nargs="+")
parser.add_argument('--n', type=int)
parser.add_argument('--iters', type=int, default=1)
parser.add_argument('--subsampling', type=int, default=True)
parser.add_argument('--jobid', type=int)

args = parser.parse_args()
debug = args.debug
if debug:
    args.num_subsample = [2, 3, 4]
    args.num_repeat = [4, 6]
    # args.b = [7, 8]
    # args.n = 15
    # args.noise_sd = np.logspace(-3.7, -4.3, num=3)
    args.b = [2, 3, 4]
    args.n = 8
    args.noise_sd = np.logspace(-2, -3.5, num=5)
    args.iters = 1
    args.jobid = "debug-" + str(uuid.uuid1())[:8]
    args.subsampling = True
    exp_dir = Path(f"results/{str(args.jobid)}")
else:
    exp_dir = Path(f"/global/scratch/users/erginbas/qspright/rna-exp-results/{str(args.jobid)}")

print(exp_dir)

print("Parameters :", args, flush=True)

query_args = {
    "query_method": "complex",
    "delays_method": "nso",
    "num_subsample": max(args.num_subsample),
    "num_repeat": max(args.num_repeat),
    "b": max(args.b),
    "all_bs": args.b
}

methods = ["qspright", "lasso"]
colors = ["red", "blue", "green", "purple"]

test_args = {
    "n_samples": 50000
}

print("Loading/Calculating data...", flush=True)

exp_dir.mkdir(parents=True, exist_ok=True)
(exp_dir / "figs").mkdir(exist_ok=True)

print(exp_dir)

helper = RNAHelper(args.n, baseline_methods=methods, subsampling=args.subsampling,
                   query_args=query_args, test_args=test_args, exp_dir=exp_dir)
n = helper.n
q = 4
print("n = {}, N = {:.2e}".format(n, q ** n))

print("Starting the tests...", flush=True)

fig, ax = plt.subplots()

for m in range(len(methods)):
    # Test QSPRIGHT with different parameters
    # Construct a grid of parameters. For each entry, run multiple test rounds.
    # Compute the average for each parameter selection.
    results_df = run_tests(methods[m], helper, args.iters, args.num_subsample, args.num_repeat,
                           args.b, args.noise_sd, parallel=False)

    # results_df.to_csv(f'results/{str(args.jobid)}/results_df_{methods[m]}.csv')

    means = results_df.groupby(["num_subsample", "num_repeat", "b", "noise_sd"], as_index=False).mean()
    stds = results_df.groupby(["num_subsample", "num_repeat", "b", "noise_sd"], as_index=False).std()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(results_df)

    x_values = []
    y_values = []

    labels = []
    all_points = []

    for i in means.index:
        mean_row = means.iloc[i]
        std_row = stds.iloc[i]
        ax.errorbar(mean_row['n_samples'], mean_row['nmse'],
                    xerr=std_row['n_samples'], yerr=std_row['nmse'], fmt="o", color=colors[m])
        all_points.append([mean_row['n_samples'], mean_row['nmse']])
        label = f'({int(mean_row["b"])},{int(mean_row["num_subsample"])},{int(mean_row["num_repeat"])})'
        labels.append(label)

    for i in range(len(all_points)):
        ax.annotate(labels[i], xy=all_points[i], xycoords='data',
                    xytext=(20, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    shrinkA=0, shrinkB=5,
                                    connectionstyle="arc3,rad=0.4",
                                    color='blue'), )

    try:
        if len(all_points) > 3:
            bcue = best_convex_underestimator(np.array(all_points))
            ax.plot(bcue[:, 0], bcue[:, 1], 'r--', lw=1.5, label="Best Cvx Underest.")
    except:
        pass

ax.set_xlabel('Total Samples')
ax.set_ylabel('Test NMSE')
plt.legend()
plt.grid()
plt.savefig(exp_dir / f'figs/nmse-vs-sample.png')
plt.show()