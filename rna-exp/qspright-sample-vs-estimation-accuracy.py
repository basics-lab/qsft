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
from rna_transform.rna_tests import run_accuracy_tests

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--num_subsample', type=int, nargs="+", default=[4])
parser.add_argument('--num_random_delays', type=int, nargs="+", default=[10])
parser.add_argument('--b', type=int, nargs="+", default=[7, 8])
parser.add_argument('--noise_sd', type=float, nargs="+", default=np.logspace(-4, -6, num=5))
parser.add_argument('--n', type=int, nargs="+", default=20)
parser.add_argument('--iters', type=int, nargs="+", default=1)
parser.add_argument('--jobid', type=int, nargs="+", default="0")

args = parser.parse_args()
debug = args.debug
if debug:
    args.num_subsample = [1]
    args.num_random_delays = [1]
    args.b = [6]
    args.n = 10
    args.iters = 2
    args.jobid = args.jobid

num_subsample_list = args.num_subsample
num_random_delays_list = args.num_random_delays
b_list = args.b
noise_sd_list = args.noise_sd
n = args.n
iters = args.iters
jobid = args.jobid

Path(f"./results/{str(jobid)}").mkdir(exist_ok=True)

print("Parameters :", args)

np.random.seed(123)

positions = np.sort(np.random.choice(50, size=n, replace=False))
query_args = {
    "query_method": "complex",
    "delays_method": "nso",
    "num_subsample": max(num_subsample_list),
    "num_random_delays": max(num_random_delays_list),
    "b": max(b_list)
}

print("positions: ", positions)
helper = RNAHelper(positions, subsampling=True, jobid=jobid, query_args=query_args)
n = helper.n
q = helper.q

print("n = {}, N = {:.2e}".format(n, q ** n))

# ## Test QSPRIGHT with different parameters
# 
# Construct a grid of parameters. For each entry, run multiple test rounds. Compute the average for each parameter selection.
results_df = run_accuracy_tests(helper, iters, num_subsample_list, num_random_delays_list, b_list, noise_sd_list, parallel=False)

means = results_df.groupby(["num_subsample", "num_random_delay", "b", "noise_sd"], as_index=False).mean()
stds = results_df.groupby(["num_subsample", "num_random_delay", "b", "noise_sd"], as_index=False).std()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(means)
print(stds)


exit()

all_points = np.zeros(shape=[0, 2])

for i, b in enumerate(b_list):
    s_values = np.mean(unique_sample_ratios[:, :, i, :], axis=-1).flatten()
    mse_values = np.mean(accs[:, :, i, :], axis=-1).flatten()
    s_std = np.std(unique_sample_ratios[:, :, i, :], axis=-1).flatten()
    mse_std = np.std(accs[:, :, i, :], axis=-1).flatten()

    plt.errorbar(s_values, mse_values, xerr=s_std, yerr=mse_std, label="b = {}".format(b), fmt="o")

    all_points = np.concatenate((all_points, np.array([s_values, mse_values]).T), axis=0)

if len(all_points) > 3:
    bcue = best_convex_underestimator(all_points)
    plt.plot(bcue[:, 0], bcue[:, 1], 'r--', lw=1.5, label="Best Cvx Underest.")

plt.xlabel('Unique Sample Ratio')
plt.ylabel('Prediction NMSE')
plt.legend()
plt.grid()
plt.savefig(f'results/{jobid}/figs/acc-vs-unique-sample-ratio.png')
plt.show()

all_points = np.zeros(shape=[0, 2])

for i, b in enumerate(b_list):
    s_values = np.mean(sample_ratios[:, :, i, :], axis=-1).flatten()
    mse_values = np.mean(accs[:, :, i, :], axis=-1).flatten()
    s_std = np.std(sample_ratios[:, :, i, :], axis=-1).flatten()
    mse_std = np.std(accs[:, :, i, :], axis=-1).flatten()

    plt.errorbar(s_values, mse_values, xerr=s_std, yerr=mse_std, label="b = {}".format(b), fmt="o")

    all_points = np.concatenate((all_points, np.array([s_values, mse_values]).T), axis=0)

if len(all_points) > 3:
    bcue = best_convex_underestimator(all_points)
    plt.plot(bcue[:, 0], bcue[:, 1], 'r--', lw=1.5, label="Best Cvx Underest.")

plt.xlabel('Total Sample Ratio')
plt.ylabel('Prediction NMSE')
plt.legend()
plt.grid()
plt.savefig(f'results/{jobid}/figs/acc-vs-total-sample-ratio.png')
plt.show()
