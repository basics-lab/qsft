import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib import ticker
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 504)
pd.set_option('display.width', 1000)

sys.path.append("..")
sys.path.append("../src")

from pathlib import Path

if __name__ == '__main__':

    font = {'family': 'sans',
            'size': 12}

    matplotlib.rc('font', **font)

    exp_dir = Path(f"results/debug-510ff35e")

    results_df = pd.read_pickle(exp_dir / "result.pkl")
    group_by = ["method", "n", "q", "num_subsample", "num_repeat", "b", "noise_sd"]
    means = results_df.groupby(group_by, as_index=False).mean()
    stds = results_df.groupby(group_by, as_index=False).std()

    print(means)

    q = results_df["q"][0]

    # fig, ax = plt.subplots(2, 2)
    colorMap = plt.get_cmap('cividis_r')
    sampleAlpha = 0.6
    timeAlpha = 0.8

    methods = ["qsft_coded"]
    method_names = ["q-SFT Coded"]

    sample_comp = [[] for _ in methods]
    time_comp = [[] for _ in methods]

    for i in means.index:
        mean_row = means.iloc[i]
        m = np.where(mean_row["method"] == np.array(methods))[0][0]
        if m != -1:
            sample_comp[m].append((mean_row["n"], mean_row["n_samples"], mean_row["nmse"]))
            time_comp[m].append((mean_row["n"], mean_row["runtime"], mean_row["nmse"]))

    min_samples = np.min(np.array(sample_comp[0]), axis=0)[1]
    max_samples = np.max(np.array(sample_comp[0]), axis=0)[1]
    sample_bin_count = 10
    ns = np.unique(np.array(sample_comp[0])[:, 0])
    sample_bins = np.linspace(np.log10(min_samples), np.log10(max_samples), sample_bin_count + 1)[1:]

    (exp_dir / "figs").mkdir(exist_ok=True)

    for m in range(len(methods)):

        fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=300)

        sample_comp_m = np.array(sample_comp[m])
        sample_bin_totals = np.zeros((len(ns), sample_bin_count))
        sample_bin_counts = np.zeros((len(ns), sample_bin_count))

        for row in sample_comp_m:
            sample_bin = np.where(sample_bins >= np.log10(row[1]))[0][0]
            n_bin = np.where(int(row[0]) == ns)[0][0]
            sample_bin_totals[n_bin, sample_bin] += row[2]
            sample_bin_counts[n_bin, sample_bin] += 1

        sample_bin_avg = sample_bin_totals / sample_bin_counts

        for n_idx in range(len(ns)):
            sample_row = sample_bin_avg[n_idx]
            if len(np.where(~np.isnan(sample_row))[0]) > 0:
                first_non_nan = np.where(~np.isnan(sample_row))[0][0]
                sample_row[:first_non_nan] = sample_row[first_non_nan]
                last_non_nan = np.where(~np.isnan(sample_row))[0][-1]
                sample_row[last_non_nan+1:] = sample_row[last_non_nan]
                sample_bin_avg[n_idx] = sample_row

        sample_bin_avg = sample_bin_avg.T
        sample_bin_avg = np.minimum(sample_bin_avg, 1)

        masked_sample_bin_avg = np.ma.array(sample_bin_avg, mask=np.isnan(sample_bin_avg))
        colorMap.set_bad('lightgrey')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        data = ax.pcolormesh(ns, 10 ** sample_bins, masked_sample_bin_avg, cmap=colorMap)
        ax.set_yscale("log")
        ax.set_xlabel("n\nN")
        ax.set_ylabel('Sample Complexity')
        ax.set_xticks(ns[1::3])
        # ax.set_yticks([10, 10**2, 10**3, 10**4, 10**5])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(rf"${{%d}}$"))
        ax.xaxis.set_label_coords(-0.03, -.04)
        secax = ax.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
        secax.xaxis.set_major_formatter(ticker.FormatStrFormatter("".join(["\n", rf"${q}^{{%d}}$"])))
        secax.set_xticks(ns[1::3])

        cbar = fig.colorbar(data, cax=cax, orientation='vertical')
        cbar.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_label('NMSE', rotation=270, labelpad=15)

        plt.tight_layout()

        plt.savefig(exp_dir / f'figs/complexity-vs-n-{methods[m]}.pdf', bbox_inches='tight',
                    transparent="True", pad_inches=0)
        plt.show()

    _, ax = plt.subplots(1, 1, figsize=(4, 3.2), dpi=300)

    for m in range(len(methods)):
        time_comp_m = np.array(time_comp[m])
        total_success_time = np.zeros(len(ns))
        count_success_time = np.zeros(len(ns))
        for row in time_comp_m:
            if row[2] < 0.1:
                n_bin = np.where(int(row[0]) == ns)[0][0]
                total_success_time[n_bin] += row[1]
                count_success_time[n_bin] += 1

        avg_success_time = total_success_time / count_success_time

        ax.plot(ns, avg_success_time, "o-", label=method_names[m])
        ax.set_xlabel("n\nN")
        ax.set_ylabel('Runtime Complexity (sec)', fontsize=11)
        ax.set_yscale("log")
        ax.grid(True)
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.set_xticks(ns[1::3])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(rf"${{%d}}$"))
        ax.xaxis.set_label_coords(-0.03, -.04)
        secax = ax.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
        secax.xaxis.set_major_formatter(ticker.FormatStrFormatter("".join(["\n", rf"${q}^{{%d}}$"])))
        secax.set_xticks(ns[1::3])

        ax.legend()

    plt.tight_layout()
    plt.savefig(exp_dir / f'figs/complexity-vs-n-runtime.pdf', bbox_inches='tight',
                transparent="True", pad_inches=0.1)
    plt.show()
