import numpy as np
import matplotlib.pyplot as plt
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

if __name__ == '__main__':

    exp_dir = Path(f"results/debug-608d6f44")

    results_df = pd.read_pickle(exp_dir / "result.pkl")
    group_by = ["method", "n", "q", "num_subsample", "num_repeat", "b", "noise_sd"]
    means = results_df.groupby(group_by, as_index=False).mean()
    stds = results_df.groupby(group_by, as_index=False).std()

    fig, ax = plt.subplots(1, 2)
    colorMap = plt.get_cmap('cool')

    for i in means.index:
        mean_row = means.iloc[i]
        ax[0].scatter(mean_row["n"], mean_row["n_samples"], c=colorMap(mean_row["nmse"]))
        ax[1].scatter(mean_row["n"], mean_row["runtime"], c=colorMap(mean_row["nmse"]))

    ax[0].set_xlabel('n')
    ax[0].set_ylabel('Sample Complexity')
    ax[0].set_yscale("log")
    ax[0].grid()

    ax[1].set_xlabel('n')
    ax[1].set_ylabel('Runtime Complexity')
    ax[1].grid()

    plt.savefig(exp_dir / f'figs/complexity-vs-n.png')
    plt.show()


