import numpy as np
import random
from tqdm import tqdm
import json
from sklearn.linear_model import Lasso
from multiprocessing import Pool
from pathlib import Path
import RNA

from src.qspright.utils import lasso_decode
from src.qspright.input_signal import Signal
from src.qspright.qspright import QSPRIGHT
from src.qspright.utils import gwht, dec_to_qary_vec, binary_ints, save_data, load_data, NpEncoder
from src.rna_transform.input_rna_signal import RnaSignal
from src.rna_transform.input_rna_signal_subsampled import RnaSubsampledSignal
from src.rna_transform.rna_utils import get_rna_base_seq


class RNAHelper:
    mfe_base = 0

    @staticmethod
    def set_mfe_base(x):
        global mfe_base
        mfe_base = x

    @staticmethod
    def _calc_data_inst(args):
        if type(args) == tuple:
            index, full = args
            fc = RNA.fold_compound(full)
            (_, mfe) = fc.mfe()
            return index, mfe - mfe_base
        else:
            (_, mfe) = RNA.fold(args)
            return mfe - mfe_base

    def __init__(self, n, subsampling=False, jobid=0, query_args=None, test_args=None):

        self.q = 4
        self.jobid = jobid

        self.exp_dir = Path(f"results/{str(self.jobid)}")

        config_path = self.exp_dir / "config.json"
        config_exists = config_path.is_file()

        if config_exists:
            with open(config_path) as f:
                config_dict = json.load(f)
            query_args = config_dict["query_args"]
        else:
            positions = list(np.sort(np.random.choice(len(get_rna_base_seq()), size=n, replace=False)))
            config_dict = {"base_seq": get_rna_base_seq(), "positions": positions, "query_args": query_args}
            with open(config_path, "w") as f:
                json.dump(config_dict, f, cls=NpEncoder)

        self.positions = config_dict["positions"]
        self.base_seq = config_dict["base_seq"]
        self.n = len(self.positions)
        self.subsampling = subsampling

        (_, mfe_base) = RNA.fold("".join(self.base_seq))
        self.set_mfe_base(mfe_base)
        self.sampling_function = self._calc_data_inst

        print("Positions: ", self.positions)
        print("Sampling Query: ", query_args)

        self.load_train_data(query_args)
        print("Training data loaded.", flush=True)
        self.load_test_data(test_args)
        print("Test data loaded.", flush=True)

    def load_train_data(self, query_args):
        """
        Constructs and saves the data corresponding to the quasi-empirical RNA fitness function
        of the Hammerhead ribozyme HH9.
        """
        if self.subsampling:
            query_args["subsampling_method"] = "qspright"
            self.rna_signal = RnaSubsampledSignal(n=self.n,
                                                  q=self.q,
                                                  query_args=query_args,
                                                  base_seq=self.base_seq,
                                                  positions=self.positions,
                                                  sampling_function=self.sampling_function,
                                                  folder=self.exp_dir / "train")
        else:
            self.rna_signal = RnaSignal(n=self.n,
                                        q=self.q,
                                        positions=self.positions,
                                        base_seq=self.base_seq,
                                        sampling_function=self.sampling_function,
                                        folder=self.exp_dir / "train")

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

        if verbosity >= 1:
            print("Finding GWHT coefficients with QSPRIGHT")

        qspright = QSPRIGHT(
            reconstruct_method="nso",
            num_subsample=num_subsample,
            num_random_delays=num_random_delays,
            b=b,
            noise_sd=noise_sd
        )

        out = qspright.transform(self.rna_signal, verbosity=verbosity, timing_verbose=True, report=report)

        if verbosity >= 1:
            print("Found GWHT coefficients")

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
                signal = RnaSignal(n=n, q=q, noise_sd=noise_sd, base_seq=self.base_seq,
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
            y = self.rna_signal
            y_oh = self.convert_onehot(y)
            n = len(self.positions)
            q = 4

            if verbose:
                print("Finding WHT coefficients with SPRIGHT")

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
        if len(beta.keys()) > 0:
            test_signal = self.test_signal.signal_t
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            batch_size = 10000

            beta_keys = list(beta.keys())
            beta_values = list(beta.values())

            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = dec_to_qary_vec(sample_idx_dec_batch, self.q, self.n)
                freqs = np.array(sample_idx_batch).T @ np.array(beta_keys).T
                H = np.exp(2j * np.pi * freqs / self.q)
                y_hat.append(H @ np.array(beta_values))

            y_hat = np.concatenate(y_hat)

            return np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
        else:
            return 1

    def load_test_data(self, test_args=None):

        (self.exp_dir / "test").mkdir(exist_ok=True)
        test_signal_t_path = self.exp_dir / "test/signal_t.pickle"

        if not test_args:
            test_args = {}

        if self.subsampling:
            query_args = {"subsampling_method": "uniform", "n_samples": test_args.get("n_samples", 50000)}
            self.test_signal = RnaSubsampledSignal(n=self.n,
                                                   q=self.q,
                                                   query_args=query_args,
                                                   base_seq=self.base_seq,
                                                   positions=self.positions,
                                                   sampling_function=self.sampling_function,
                                                   folder=self.exp_dir / "test")
        else:
            self.test_signal = self.rna_signal
