import numpy as np
import RNA
import itertools
from tqdm import tqdm
from sklearn.linear_model import Lasso
from multiprocessing import Pool
import rna_transform.utils as utils
from qspright.inputsignal import Signal
from qspright.qspright_nso import QSPRIGHT
from qspright.utils import gwht, dec_to_qary_vec, binary_ints
from rna_transform.input_rna_signal import SignalRNA
from rna_utils import  insert, dna_to_rna, get_rna_base_seq, _calc_data_inst


class RNAHelper:
    def __init__(self, positions):
        self.positions = positions
        self.base_seq = dna_to_rna(get_rna_base_seq())
        self.n = len(positions)
        self.q = 4
        self.rna_data = self.load_rna_data()


    def calculate_rna_data(self, verbose = False, parallel = True):
        """
        Constructs and saves the data corresponding to the quasi-empirical RNA fitness function
        of the Hammerhead ribozyme HH9.
        """

        nucs = ["A", "U", "C", "G"]
        seqs_as_list = list(itertools.product(nucs, repeat=len(self.positions)))
        seqs = ["".join(s) for s in seqs_as_list]

        print("Calculating free energies...")

        if parallel:

            with Pool() as pool:
                y = list(tqdm(pool.imap(_calc_data_inst,
                                        zip(itertools.repeat(self.base_seq), itertools.repeat(self.positions), seqs)),
                              total=len(seqs)))

        else:
            y = []

            for s in tqdm(seqs):
                full = insert(self.base_seq, self.positions, s)
                (ss, mfe) = RNA.fold(full)
                y.append(mfe)

        np.save("results/rna_data.npy", np.array(y))

        return


    def load_rna_data(self, centering=True):
        try:
            y = np.load("results/rna_data.npy")
            # print(np.mean(y))
            if centering:
                return y - np.mean(y)
            else:
                return y
        except:
            return None


    def get_rna_data(self):
        if self.rna_data is None:
            raise RuntimeError("RNA data is not yet computed.")
        else:
            return self.rna_data


    def generate_householder_matrix(self):
        nucs = ["A", "U", "C", "G"]
        positions = self.positions

        nucs_idx = {nucs[i]: i for i in range(len(nucs))}
        seqs_as_list = list(itertools.product(nucs, repeat=len(positions)))

        int_seqs = [[nucs_idx[si] for si in s] for s in seqs_as_list]

        print("Constructing Fourier matrix...")
        X = utils.fourier_from_seqs(int_seqs, [4] * self.n)

        return X

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
            y = self.get_rna_data()
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
            y = self.get_rna_data()
            beta = gwht(y, q=4, n=n)
            print("Found GWHT coefficients")
            if save:
                np.save("results/rna_beta_gwht.npy", beta)
            return beta

    def _calculate_rna_qspright(self, save=False, report = False, noise_sd=None, verbose = False, num_subsample = 4, num_random_delays = 10, b = None, on_demand_comp=False):
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
            y = self.get_rna_data()
            n = len(self.positions)
            q = 4
            if verbose:
                print("Finding GWHT coefficients with QSPRIGHT")

            if noise_sd is None:
                noise_sd = 300 / (q ** n)

            if on_demand_comp:
                signal = SignalRNA(n=n, q=q, noise_sd=noise_sd, base_seq=self.base_seq,
                                   positions=self.positions, parallel=True)
            else:
                signal = Signal(n=n, q=q, signal=y, noise_sd=noise_sd)

            spright = QSPRIGHT(
                query_method="complex",
                delays_method="nso",
                reconstruct_method="nso",
                num_subsample = num_subsample,
                num_random_delays = num_random_delays,
                b = b
            )

            out = spright.transform(signal, verbose=False, report=report)
            if report:
                beta, n_used, peeled = out
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
            y = self.get_rna_data()
            y = self.load_rna_data()
            y_oh = self.convert_onehot(y)
            beta = gwht(y_oh, q=2, n=self.n*self.q)
            print("Found one-hot WHT coefficients")
            if save:
                np.save("results/rna_beta_onehot_wht.npy", beta)
            return beta


    def _calculate_rna_onehot_spright(self, save=False, report = False, noise_sd=None, verbose = False, num_subsample = 4, num_random_delays = 10, b = None):
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

            signal = Signal(n=n*q, q=2, signal=y_oh, noise_sd=noise_sd)
            spright = QSPRIGHT(
                query_method="complex",
                delays_method="nso",
                reconstruct_method="nso",
                num_subsample = num_subsample,
                num_random_delays = num_random_delays,
                b = b
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

    #def _test_rna_qspright(self):