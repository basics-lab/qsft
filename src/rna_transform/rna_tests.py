import itertools
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
import pandas as pd

tqdm = partial(tqdm, position=0, leave=True)

def _test(i):
    df_row = test_df.iloc[i]
    num_subsample, num_random_delays, b, noise_sd = int(df_row["num_subsample"]), int(df_row["num_random_delay"]), int(df_row["b"]), df_row["noise_sd"]

    # set model arguments
    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_random_delays"] = num_random_delays
    model_kwargs["b"] = b
    model_kwargs["noise_sd"] = noise_sd
    model_kwargs["report"] = True
    model_kwargs["verbosity"] = 0

    spright_result = helper_obj.compute_rna_model(method="qspright", **model_kwargs)

    test_kwargs = {}
    test_kwargs["beta"] = spright_result.get("gwht")

    del spright_result['gwht']
    del spright_result['locations']

    nmse = helper_obj.test_rna_model(method="qspright", **test_kwargs)

    result = {}
    result["ratio_samples"] = spright_result.get("n_samples") / helper_obj.q ** helper_obj.n
    result["max_hamming_weight"] = spright_result.get("max_hamming_weight")
    result["nmse"] = nmse

    return result


def run_tests(helper, iters, num_subsample_list, num_random_delays_list, b_list, noise_sd_list, parallel=True):

    global test_df
    global helper_obj

    helper_obj = helper

    test_params_list = list(itertools.product(num_subsample_list, num_random_delays_list, b_list, noise_sd_list, range(iters)))
    test_df = pd.DataFrame(data=test_params_list, columns=["num_subsample", "num_random_delay", "b", "noise_sd", "iter"])

    exp_count = len(test_df)

    if parallel:
        with Pool() as pool:
            # run the tests in parallel
            pred = list(tqdm(pool.imap(_test, range(exp_count)), total=exp_count))
    else:
        pred = []
        for i in tqdm(range(exp_count)):
            pred.append(_test(i))

    results_df = pd.DataFrame(data = pred)

    results_df = pd.concat([test_df, results_df], axis=1, join="inner")

    return results_df