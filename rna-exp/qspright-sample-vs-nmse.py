#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

sys.path.append("..")
sys.path.append("../src")

from rna_transform.rna_helper import RNAHelper
from qspright.utils import best_convex_underestimator
from rna_transform.rna_tests import run_tests
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--num_subsample', type=int, nargs="+")
parser.add_argument('--num_random_delays', type=int, nargs="+")
parser.add_argument('--b', type=int, nargs="+")
parser.add_argument('--noise_sd', type=float, nargs="+")
parser.add_argument('--n', type=int)
parser.add_argument('--iters', type=int, default=1)
parser.add_argument('--subsampling', type=int, default=True)
parser.add_argument('--jobid', type=int)

args = parser.parse_args()
debug = args.debug
if debug:
    args.num_subsample = [1, 2, 3]
    args.num_random_delays = [1, 2, 3, 4, 5]
    args.b = [6, 7]
    args.n = 10
    args.noise_sd = np.logspace(-2.5, -3, num=1)
    args.iters = 1
    args.jobid = 888
    args.subsampling = True

num_subsample_list = args.num_subsample
num_random_delays_list = args.num_random_delays
b_list = args.b
noise_sd_list = args.noise_sd
n = args.n
iters = args.iters
jobid = args.jobid
subsampling = args.subsampling

Path(f"./results/{str(jobid)}").mkdir(exist_ok=True)
Path(f"./results/{str(jobid)}/figs").mkdir(exist_ok=True)

print("Parameters :", args, flush=True)

query_args = {
    "query_method": "complex",
    "delays_method": "nso",
    "num_subsample": max(num_subsample_list),
    "num_random_delays": max(num_random_delays_list),
    "b": max(b_list),
    "all_bs": b_list
}

baseline_methods = []

test_args = {
    "n_samples": 500000
}

print("Loading/Calculating data...", flush=True)

helper = RNAHelper(n, baseline_methods=baseline_methods, subsampling=subsampling,
                   jobid=jobid, query_args=query_args, test_args=test_args)
n = helper.n
q = 4
print("n = {}, N = {:.2e}".format(n, q ** n))

print("Starting the tests...", flush=True)

# Test QSPRIGHT with different parameters
# Construct a grid of parameters. For each entry, run multiple test rounds.
# Compute the average for each parameter selection.
results_df = run_tests(helper, iters, num_subsample_list, num_random_delays_list,
                       b_list, noise_sd_list, parallel=True)

means = results_df.groupby(["num_subsample", "num_random_delay", "b", "noise_sd"], as_index=False).mean()
stds = results_df.groupby(["num_subsample", "num_random_delay", "b", "noise_sd"], as_index=False).std()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(results_df)

all_points = []

for i in means.index:
    mean_row = means.iloc[i]
    std_row = stds.iloc[i]
    plt.errorbar(mean_row['ratio_samples'], mean_row['nmse'],
                 xerr=std_row['ratio_samples'], yerr=std_row['nmse'], fmt="o")

    all_points.append([mean_row['ratio_samples'], mean_row['nmse']])

try:
    if len(all_points) > 3:
        bcue = best_convex_underestimator(np.array(all_points))
        plt.plot(bcue[:, 0], bcue[:, 1], 'r--', lw=1.5, label="Best Cvx Underest.")
except:
    pass

plt.xlabel('Unique Sample Ratio')
plt.ylabel('Prediction NMSE')
plt.legend()
plt.grid()
plt.savefig(f'results/{str(jobid)}/figs/acc-vs-unique-sample-ratio.png')
plt.show()

results_df.to_csv(f'results/{str(jobid)}/results_df.csv')
