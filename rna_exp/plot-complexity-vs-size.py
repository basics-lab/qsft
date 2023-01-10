import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy import interpolate

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 504)
pd.set_option('display.width', 1000)

sys.path.append("..")
sys.path.append("../src")

from pathlib import Path
from qsft.utils import best_convex_underestimator
import scipy

if __name__ == '__main__':

    exp_nums = ["13844137", "13844138", "13846802", "13846805"]
    dfs = []
    for exp_id in exp_nums:
        exp_dir = Path(f"results_final/{exp_id}")
        dfs.append(pd.read_pickle(exp_dir / "result.pkl"))

    results_df = pd.concat(dfs, ignore_index=True)

    group_by = ["method", "n", "q", "num_subsample", "num_repeat", "b"]
    results_df = results_df.groupby(group_by, as_index=False).min()

    # fig, ax = plt.subplots(2, 2)
    colorMap = plt.get_cmap('cividis_r')
    sampleAlpha = 0.6
    timeAlpha = 0.8

    methods = ["qsft"]

    _, ax = plt.subplots(1, 1, figsize=(4, 3))

    ns = results_df["n"].unique()

    for n in ns:
        df_n = results_df.loc[results_df['n'] == n]
        xs = df_n["n_samples"] / 10
        ys = 10 * df_n["nmse"] / np.log(df_n["n_samples"])
        all_points = list(zip(xs, ys))
        model = lambda t, a, b: a + b * np.exp(-t)
        a, _ = scipy.optimize.curve_fit(model, np.log10(xs), ys)
        polyline_points = np.logspace(2.6, 7, 50)
        plt.plot(polyline_points, [model(np.log10(p), a[0], a[1]) for p in polyline_points])
        # if len(all_points) > 3:
        #     bcue = best_convex_underestimator(np.array(all_points))
        #     ax.plot(bcue[:, 0], bcue[:, 1], lw=1.5, label=f"n={n}")
        ax.scatter(*zip(*all_points), s=10)

    ax.set_xscale("log")
    ax.set_xlabel('Sample Complexity')
    ax.set_ylabel('Test NMSE')
    ax.legend()
    ax.grid()
    # ax.set_xticks(ns)

    # for m in range(len(methods)):
    #     time_comp_m = np.array(time_comp[m])
    #     total_success_time = np.zeros(len(ns))
    #     count_success_time = np.zeros(len(ns))
    #     for row in time_comp_m:
    #         if row[2] < 0.1:
    #             n_bin = np.where(int(row[0]) == ns)[0][0]
    #             total_success_time[n_bin] += row[1]
    #             count_success_time[n_bin] += 1
    #
    #     avg_success_time = total_success_time / count_success_time
    #
    #     print(total_success_time)
    #
    #     ax[-1].plot(ns, avg_success_time, "o-")
    #     ax[-1].set_xlabel('n')
    #     ax[-1].set_ylabel('Runtime Complexity (sec)')
    #     ax[-1].set_yscale("log")
    #     ax[-1].grid(True)
    #     ax[-1].set_xticks(ns)

    plt.tight_layout()
    Path(f"figs/").mkdir(exist_ok=True)
    plt.savefig('figs/complexity-vs-n-rna.pdf', bbox_inches='tight',
                transparent="True", pad_inches=0.1)
    plt.show()


