import itertools
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import numpy as np
from contextlib import closing

def _nmse_test(i):
    param_idx = i % len(test_params_list)
    test_params = test_params_list[param_idx]
    num_subsample, num_random_delays, b = test_params

    # set model arguments
    model_kwargs = {}
    model_kwargs["save"] = False
    model_kwargs["noise_sd"] = 800 / (helper_obj.q ** helper_obj.n)
    model_kwargs["report"] = True
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_random_delays"] = num_random_delays
    model_kwargs["b"] = b
    model_kwargs["on_demand_comp"] = True

    gwht, (n_used, n_used_unique, used_unique), peeled = helper_obj.compute_rna_model(method="qspright", **model_kwargs)
    emp_beta_qspright = np.reshape(gwht, -1)

    #     print("done: ", num_subsample, num_random_delays, b)

    # calculate metrics
    sample_ratio = n_used / helper_obj.q ** helper_obj.n
    unique_sample_ratio = n_used_unique / helper_obj.q ** helper_obj.n
    nmse = np.sum(np.abs(emp_beta_gwht - emp_beta_qspright) ** 2) / np.sum(np.abs(emp_beta_gwht) ** 2)

    # print(len(peeled), nmse)
    return param_idx, sample_ratio, unique_sample_ratio, nmse

def run_nmse_tests(helper, iters, num_subsample_list, num_random_delays_list, b_list, ground_truth):

    global test_params_list
    global helper_obj
    global emp_beta_gwht

    helper_obj = helper
    test_params_list = list(itertools.product(num_subsample_list, num_random_delays_list, b_list))
    emp_beta_gwht = ground_truth

    exp_count = iters * len(test_params_list)

    with Pool() as pool:
        # run the tests in parallel
        pred = list(tqdm(pool.imap(_gwht_test, range(exp_count)), total=exp_count))

    test_params_idx_list = list(
        itertools.product(np.arange(len(num_subsample_list)), np.arange(len(num_random_delays_list)),
                          np.arange(len(b_list))))

    sample_ratios = np.zeros((len(num_subsample_list), len(num_random_delays_list), len(b_list), iters))
    unique_sample_ratios = np.zeros((len(num_subsample_list), len(num_random_delays_list), len(b_list), iters))
    nmses = np.zeros((len(num_subsample_list), len(num_random_delays_list), len(b_list), iters))

    for i, exp_result in enumerate(pred):
        iter_idx = i // len(test_params_list)
        test_params_idx = i % len(test_params_list)
        test_params_idx = test_params_idx_list[test_params_idx]
        param_idx, sample_ratio, unique_sample_ratio, nmse = exp_result
        idx = test_params_idx + (iter_idx,)
        sample_ratios[idx] = sample_ratio
        unique_sample_ratios[idx] = unique_sample_ratio
        nmses[idx] = nmse

    return sample_ratios, unique_sample_ratios, nmses


def _acc_test(i):
    param_idx = i % len(test_params_list)
    test_params = test_params_list[param_idx]
    num_subsample, num_random_delays, b = test_params

    # set model arguments
    model_kwargs = {}
    model_kwargs["save"] = False
    model_kwargs["noise_sd"] = 800 / (helper_obj.q ** helper_obj.n)
    model_kwargs["report"] = True
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_random_delays"] = num_random_delays
    model_kwargs["b"] = b
    model_kwargs["on_demand_comp"] = True

    gwht, (n_used, n_used_unique, used_unique), peeled = helper_obj.compute_rna_model(method="qspright", **model_kwargs)

    #     print("done: ", num_subsample, num_random_delays, b)

    # calculate metrics
    sample_ratio = n_used / helper_obj.q ** helper_obj.n
    unique_sample_ratio = n_used_unique / helper_obj.q ** helper_obj.n

    test_kwargs = {}
    test_kwargs["beta"] = gwht
    test_kwargs["on_demand_comp"] = False

    acc = helper_obj.test_rna_model(method="qspright", **test_kwargs)
    return param_idx, sample_ratio, unique_sample_ratio, acc


def run_accuracy_tests(helper, iters, num_subsample_list, num_random_delays_list, b_list, ground_truth, parallel=False):

    global test_params_list
    global helper_obj
    global emp_beta_gwht

    helper_obj = helper
    test_params_list = list(itertools.product(num_subsample_list, num_random_delays_list, b_list))
    emp_beta_gwht = ground_truth

    exp_count = iters * len(test_params_list)

    if parallel:
        with Pool() as pool:
            # run the tests in parallel
            pred = list(tqdm(pool.imap(_acc_test, range(exp_count)), total=exp_count))
    else:
        pred = []
        for i in tqdm(range(exp_count)):
            pred.append(_acc_test(i))

    test_params_idx_list = list(
        itertools.product(np.arange(len(num_subsample_list)), np.arange(len(num_random_delays_list)),
                          np.arange(len(b_list))))

    sample_ratios = np.zeros((len(num_subsample_list), len(num_random_delays_list), len(b_list), iters))
    unique_sample_ratios = np.zeros((len(num_subsample_list), len(num_random_delays_list), len(b_list), iters))
    nmses = np.zeros((len(num_subsample_list), len(num_random_delays_list), len(b_list), iters))

    for i, exp_result in enumerate(pred):
        iter_idx = i // len(test_params_list)
        test_params_idx = i % len(test_params_list)
        test_params_idx = test_params_idx_list[test_params_idx]
        param_idx, sample_ratio, unique_sample_ratio, nmse = exp_result
        idx = test_params_idx + (iter_idx,)
        sample_ratios[idx] = sample_ratio
        unique_sample_ratios[idx] = unique_sample_ratio
        nmses[idx] = nmse

    return sample_ratios, unique_sample_ratios, nmses