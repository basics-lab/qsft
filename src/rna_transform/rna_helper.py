import numpy as np
import RNA
import itertools
import random
from tqdm import tqdm
from sklearn.linear_model import Lasso
from multiprocessing import Pool
import pickle
import sys

import src.rna_transform.utils as utils
from src.qspright.utils import lasso_decode
from src.qspright.inputsignal import Signal
from src.qspright.qspright_precomputed import QSPRIGHT
from src.qspright.utils import gwht, dec_to_qary_vec, binary_ints, qary_ints, save_data, load_data
from src.rna_transform.input_rna_signal import SignalRNA
from src.rna_transform.input_rna_signal_precomputed import PrecomputedSignalRNA
from src.rna_transform.rna_utils import insert, get_rna_base_seq, _calc_data_inst, display_top
import tracemalloc


class RNAHelper:
    def __init__(self, positions, subsampling=False, jobid = 0, query_args = {}, test_args ={}):
        tracemalloc.start()

        self.positions = positions
        self.n = len(positions)
        self.q = 4
        self.jobid = jobid

        self.load_rna_data(subsampling, query_args)
        if self.rna_signal is None:
            self.calculate_rna_data(subsampling, query_args)

        print("Training data calculated/loaded.", flush=True)

        self.load_rna_test_data()
        if self.rna_test_samples is None:
            self.calculate_test_samples(test_args)

        print("Test data calculated/loaded.", flush=True)

        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot, limit = 20)

    def calculate_rna_data(self, subsampling, query_args, verbose=False, parallel=True):
        """
        Constructs and saves the data corresponding to the quasi-empirical RNA fitness function
        of the Hammerhead ribozyme HH9.
        """
        if subsampling:
            self.rna_signal = PrecomputedSignalRNA(n=self.n,
                                                   q=self.q,
                                                   positions=self.positions,
                                                   query_args=query_args)
            self.rna_signal.subsample(keep_samples=True, save_samples_to_file=True,
                                      foldername=f"results/{str(self.jobid)}/rna_subsampled", save_all_b=False)
        else:
            self.rna_signal = SignalRNA(n=self.n, q=self.q, positions=self.positions, parallel=parallel)
            self.rna_signal.sample()

    def load_rna_data(self, subsampling, query_args):
        try:
            if subsampling:
                num_subsample = query_args.get("num_subsample")
                M_select = num_subsample * [True]
                self.rna_signal = PrecomputedSignalRNA(signal=f"results/{str(self.jobid)}/rna_subsampled", M_select = M_select)
            else:
                y = np.load(f"results/{str(self.jobid)}/rna_data.npy")
                self.rna_signal = Signal(n=self.n, q=self.q, signal=y, noise_sd=noise_sd)
        except FileNotFoundError:
            self.rna_signal = None
            return

    def get_rna_data(self):
        return self.rna_signal

    def compute_rna_model(self, method, **kwargs):
        if method == "householder":
            return self._calculate_rna_householder(**kwargs)
        elif method == "gwht":
            return self._calculate_rna_gwht(**kwargs)
        elif method == "qspright":
            return self._calculate_rna_qspright(**kwargs)
        elif method == "onehot_wht":
            return self._calculate_rna_onehot_wht(**kwargs)
        elif method == "onehot_spright":
            return self._calculate_rna_onehot_spright(**kwargs)
        else:
            raise NotImplementedError()

    def test_rna_model(self, method, **kwargs):
        if method == "householder":
            raise NotImplementedError()
        elif method == "gwht":
            raise NotImplementedError()
        elif method == "qspright":
            return self._test_rna_qspright(**kwargs)
        elif method == "onehot_wht":
            raise NotImplementedError()
        elif method == "onehot_spright":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def _calculate_rna_householder(self, save=False):
        """
        Calculates Fourier coefficients of the RNA fitness function. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        try:
            beta = np.load("results/rna_beta_lasso.npy")
            print("Loaded saved beta array.")
            return beta
        except FileNotFoundError:
            alpha = 1e-12
            y = self.rna_signal
            X = self.generate_householder_matrix()
            print("Fitting Lasso coefficients (this may take some time)...")
            model = Lasso(alpha=alpha, fit_intercept=False)
            model.fit(X, y)
            beta = model.coef_
            if save:
                np.save("results/rna_beta_lasso.npy", beta)
            return beta

    def _calculate_rna_gwht(self, save=False):
        """
        Calculates GWHT coefficients of the RNA fitness function. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        try:
            beta = np.load("results/rna_beta_gwht.npy")
            print("Loaded saved beta array for GWHT.")
            return beta
        except FileNotFoundError:
            n = self.n
            y = self.rna_signal
            beta = gwht(y, q=4, n=n)
            print("Found GWHT coefficients")
            if save:
                np.save("results/rna_beta_gwht.npy", beta)
            return beta

    def _calculate_rna_qspright(self, save=False, report=False, noise_sd=None, verbosity=0, num_subsample=4,
                                num_random_delays=10, b=None):
        """
        Calculates GWHT coefficients of the RNA fitness function using QSPRIGHT. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        try:
            beta = np.load("results/rna_beta_qspright.npy")
            print("Loaded saved beta array for GWHT QSPRIGHT.")
            return beta
        except FileNotFoundError:
            n = len(self.positions)
            q = 4
            if verbosity >= 1:
                print("Finding GWHT coefficients with QSPRIGHT")

            spright = QSPRIGHT(
                reconstruct_method="nso",
                num_subsample=num_subsample,
                num_random_delays=num_random_delays,
                b=b,
                noise_sd = noise_sd
            )

            out = spright.transform(self.rna_signal, verbosity=verbosity, timing_verbose = True, report=report)

            if verbosity >= 1:
                print("Found GWHT coefficients")
            if save:
                # TODO fix the bug here (beta is no longer an array, it is a dict)
                np.save("results/rna_beta_qspright.npy", out.get("gwht"))

            return out

    def _calculate_rna_lasso(self, save=False, report=False, noise_sd=None, verbose=False, on_demand_comp=False,
                             sampling_rate=0.1):
        """
        Calculates GWHT coefficients of the RNA fitness function using LASSO. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        try:
            beta = np.load("results/rna_beta_LASSO.npy")
            print("Loaded saved beta array for GWHT LASSO.")
            return beta
        except FileNotFoundError:
            y = self.rna_signal
            n = len(self.positions)
            q = 4
            if verbose:
                print("Finding GWHT coefficients with LASSO")

            if noise_sd is None:
                noise_sd = 300 / (q ** n)

            if on_demand_comp:
                signal = SignalRNA(n=n, q=q, noise_sd=noise_sd, base_seq=self.base_seq,
                                   positions=self.positions, parallel=True)
            else:
                signal = Signal(n=n, q=q, signal=y, noise_sd=noise_sd)

            out = lasso_decode(signal, sampling_rate)
            if report:
                beta, peeled = out
            else:
                beta = out

            if verbose:
                print("Found GWHT coefficients")
            if save:
                np.save("results/rna_beta_qspright.npy", beta)

            return out

    # def convert_onehot(y):
    #     n = len(self.positions)
    #     q = 4
    #     y_oh = np.zeros(2 ** (n * q))
    #
    #     for i in range(q ** n):
    #         i_oh = np.zeros(n * q, dtype=np.int32)
    #         i_qary = np.array(dec_to_qary_vec([i], q, n)).T[0]
    #         for loc, symbol in enumerate(i_qary):
    #             i_oh[loc * q + symbol] = 1
    #         i_oh_dec = qary_vec_to_dec(i_oh, 2)
    #         y_oh[i_oh_dec] = y[i]
    #
    #     return y_oh

    def fill_with_neighbor_mean(self, y):

        n = self.n

        idxs = binary_ints(n).T
        nan_left = True

        while nan_left:
            nan_left = False
            for idx in idxs:
                if np.isnan(y[tuple(idx)]):
                    nan_left = True
                    neighbor_values = []
                    for pos in range(n):
                        idx_temp = idx.copy()
                        idx_temp[pos] = 1 - idx_temp[pos]
                        neighbor_values.append(y[tuple(idx_temp)])
                    if np.sum(~np.isnan(neighbor_values)) > 0:
                        y[tuple(idx)] = np.nanmean(neighbor_values)

        return y

    def convert_onehot(self, y):
        n = self.n
        q = self.q

        y_oh = np.empty([2] * (n * q))
        y_oh[:] = np.nan

        for i in range(q ** n):
            i_oh = np.zeros(n * q, dtype=np.int32)
            i_qary = np.array(dec_to_qary_vec([i], q, n)).T[0]
            for loc, symbol in enumerate(i_qary):
                i_oh[loc * q + symbol] = 1
            y_oh[tuple(i_oh)] = y[i]

        y_oh_filled = self.fill_with_neighbor_mean(y_oh)

        return np.reshape(y_oh_filled, [2 ** (n * q)])

    def _calculate_rna_onehot_wht(self, save=False):
        """
        Calculates WHT coefficients of the one-hot RNA fitness function. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        try:
            beta = np.load("results/rna_beta_onehot_wht.npy")
            print("Loaded saved beta array for GWHT.")
            return beta
        except FileNotFoundError:
            y = self.rna_signal
            y = self.load_rna_data()
            y_oh = self.convert_onehot(y)
            beta = gwht(y_oh, q=2, n=self.n * self.q)
            print("Found one-hot WHT coefficients")
            if save:
                np.save("results/rna_beta_onehot_wht.npy", beta)
            return beta

    def _calculate_rna_onehot_spright(self, save=False, report=False, noise_sd=None, verbose=False, num_subsample=4,
                                      num_random_delays=10, b=None):
        """
        Calculates GWHT coefficients of the RNA fitness function using QSPRIGHT. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        try:
            beta = np.load("results/rna_beta_onehot_spright.npy")
            print("Loaded saved beta array for one-hot SPRIGHT.")
            return beta

        except FileNotFoundError:
            y = self.get_rna_data()
            y_oh = self.convert_onehot(y)
            n = len(self.positions)
            q = 4

            if verbose:
                print("Finding WHT coefficients with SPRIGHT")

            if noise_sd is None:
                noise_sd = 300 / (2 ** (q * n))

            signal = Signal(n=n * q, q=2, signal=y_oh, noise_sd=noise_sd)
            spright = QSPRIGHT(
                query_method="complex",
                delays_method="nso",
                reconstruct_method="nso",
                num_subsample=num_subsample,
                num_random_delays=num_random_delays,
                b=b
            )

            out = spright.transform(signal, verbose=False, report=report)
            if report:
                beta, n_used, peeled = out
            else:
                beta = out

            if verbose:
                print("Found GWHT coefficients")
            if save:
                np.save("results/rna_beta_onehot_spright.npy", beta)

            return out

    def _test_rna_qspright(self, beta):
        """
        :param beta:
        :return:
        """

        n = len(self.positions)
        q = 4
        N = q ** n

        if len(beta.keys())>0:
            sample_idx = self.rna_test_indices
            y = self.rna_test_samples

            freqs = np.array(sample_idx).T @ np.array(list(beta.keys())).T
            H = np.exp(2j * np.pi * freqs / q)
            y_hat = H @ np.array(list(beta.values()))
            return np.linalg.norm(y_hat - y) ** 2 / np.linalg.norm(y) ** 2
        else:
            return 1


    def calculate_test_samples(self, test_args):

        n_samples = test_args.get("n_samples", 500000)
        parallel = test_args.get("parallel", True)

        N = self.q ** self.n
        q = self.q
        n = self.n

        n_samples = np.minimum(n_samples, N)
        sample_idx = random.sample(range(N), n_samples)
        sample_idx = dec_to_qary_vec(sample_idx, q, n)

        nucs = np.array(["A", "U", "C", "G"])
        sample_idx = np.byte(np.array(sample_idx))
        mean = -21.23934478693991

        if parallel:

            query = []
            for i in range(sample_idx.shape[1]):
                seq = nucs[sample_idx[:, i]]
                full = insert(get_rna_base_seq(), self.positions, seq)
                query.append(full)

            with Pool() as pool:
                y = list(tqdm(pool.imap(_calc_data_inst, query), total=len(query), miniters=2000))

            samples = np.array(y) - mean

        else:
            y = []
            for i in tqdm(range(sample_idx.shape[1])):
                seq = ""
                for nuc_idx in sample_idx[:, i]:
                    seq = seq + nucs[nuc_idx]
                full = insert(get_rna_base_seq(), self.positions, seq)
                (ss, mfe) = RNA.fold(full)
                y.append(mfe)

            samples = np.csingle(np.array(y) - mean)

        self.rna_test_indices, self.rna_test_samples = sample_idx, samples

        save_data((sample_idx, samples), f"results/{str(self.jobid)}/rna_test.pickle")

    def load_rna_test_data(self):
        try:
            self.rna_test_indices, self.rna_test_samples = load_data(f"results/{str(self.jobid)}/rna_test.pickle")
        except FileNotFoundError:
            self.rna_test_indices, self.rna_test_samples = None, None
            return