#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("..")
sys.path.append("../src")

from rna_transform.rna_helper import RNAHelper
from qspright.utils import best_convex_underestimator
from rna_transform.rna_tests import run_accuracy_tests

np.random.seed(123)

debug = False

if debug:
    positions = np.sort(np.random.choice(50, size=10, replace=False))
    query_args = {
        "query_method": "complex",
        "delays_method": "nso",
        "num_subsample": 1,
        "num_random_delays": 1,
        "b": 6
    }
else:
    positions = np.sort(np.random.choice(50, size=20, replace=False))
    query_args = {
        "query_method": "complex",
        "delays_method": "nso",
        "num_subsample": 4,
        "num_random_delays": 10,
        "b": 8
    }

print("positions: ", positions)
helper = RNAHelper(positions, subsampling=True, query_args=query_args)
n = helper.n
q = helper.q

print("n = {}, N = {:.2e}".format(n, q ** n))


# ## Test QSPRIGHT with different parameters
# 
# Construct a grid of parameters. For each entry, run multiple test rounds. Compute the average for each parameter selection. 

if debug:
    iters = 1
    num_subsample_list = [1]
    num_random_delays_list = [1]
    b_list = [6]
else:
    iters = 10
    num_subsample_list = [2, 3, 4]
    num_random_delays_list = [4, 6, 8, 10]
    b_list = [6, 7, 8]

result = run_accuracy_tests(helper, iters, num_subsample_list, num_random_delays_list, b_list)

sample_ratios, unique_sample_ratios, accs, hamming_ws = result

for i, b in enumerate(b_list):
    print(b, np.mean(hamming_ws[:, :, i, :]))

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
plt.savefig('figs/acc-vs-unique-sample-ratio.png')
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
plt.savefig('figs/acc-vs-total-sample-ratio.png')
plt.show()


