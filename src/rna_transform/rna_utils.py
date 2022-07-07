import numpy as np
import rna
import itertools
from tqdm import tqdm
from sklearn.linear_model import Lasso
from multiprocessing import Pool
import rna_transform.utils as utils
from qspright.inputsignal import Signal
from qspright.qspright_nso import QSPRIGHT
from qspright.utils import gwht, dec_to_qary_vec, binary_ints

"""
Utility functions for loading and processing the quasi-empirical RNA fitness function.
"""


def dna_to_rna(seq):
    """
    Converts DNA sequences to RNA sequences.
    """
    rs = []
    for s in seq:
        if s == 'T':
            rs.append('U')
        else:
            rs.append(s)
    return "".join(rs)


def insert(base_seq, positions, sub_seq):
    """
    Inserts a subsequence into a base sequence
    """
    new_seq = list(base_seq)
    for i, p in enumerate(positions):
        new_seq[p-1] = sub_seq[i]
    return "".join(new_seq)


def get_rna_base_seq():
    """
    Returns the sequence of RFAM: AANN01066007.1
    """
    base_seq = "CTGAGCCGTTACCTGCAGCTGATGAGCTCCAAAAAGAGCGAAACCTGCTAGGTCCTGCAGTACTGGCTTAAGAGGCT"
    return base_seq


def sample_structures_and_find_pairs(base_seq, positions, samples=10000):
    """
    Samples secondary structures from the Boltzmann distribution
    and finds pairs of positions that are paired in any of the
    sampled strutures.
    """
    md = rna.md()
    md.uniq_ML = 1
    fc = rna.fold_compound(base_seq, md)
    (ss, mfe) = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()

    important_pairs = set()
    for s in fc.pbacktrack(10000):
        pairs = find_pairs(s)
        for p in pairs:
            if p[0] in positions and p[1] in positions:
                if p[0] > p[1]:
                    print(p, s)
                important_pairs.add(tuple(p))
    return important_pairs


def pairs_to_neighborhoods(positions, pairs):
    """
    Converts a list of pairs of interacting positions into a set of neighborhoods.
    """
    V = []
    for i, p in enumerate(positions):
        Vp = [i+1]
        for pair in pairs:
            if pair[0] == p:
                Vp.append(positions.index(pair[1]) + 1)
            elif pair[1] == p:
                Vp.append(positions.index(pair[0]) + 1)
        V.append(sorted(Vp))
    return V


def find_pairs(ss):
    """
    Finds interacting pairs in a RNA secondary structure
    """
    pairs = []
    op = []
    N = len(ss)
    for i in range(N):
        if ss[i] == '(':
            op.append(i)
        elif ss[i] == ')':
            pair = (op.pop(), i)
            pairs.append(pair)
    return pairs


def _calc_data_inst(args):
    base_seq, positions, s = args
    full = insert(base_seq, positions, s)
    (ss, mfe) = rna.fold(full)
    return mfe


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

        # construct insertion sequences
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
                (ss, mfe) = rna.fold(full)
                y.append(mfe)

        np.save("results/rna_data.npy", np.array(y))

        return


    def load_rna_data(self, centering=True):
        try:
            y = np.load("results/rna_data.npy")
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

    def calculate_rna_lasso(self, save=False):
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


    def calculate_rna_gwht(self, save=False):
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

    def calculate_rna_qspright(self, save=False, report = False, noise_sd=None, verbose = False, num_subsample = 4, num_random_delays = 10, b = None):
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


    def calculate_rna_onehot_wht(self, save=False):
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


    def calculate_rna_onehot_spright(self, save=False, report = False, noise_sd=None, verbose = False, num_subsample = 4, num_random_delays = 10, b = None):
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

'''

def calculate_rna_gnk_wh_coefficient_vars(pairs_from_scratch=False, return_neighborhoods=False):
    """
    Returns the variances of WH coefficients in GNK fitness functions with 
    Structural neighborhoods corresponding to RNA secondary structure. If pairs_from_scratch
    is True, then structures are sampled to find paired positions, otherwise pre-calculated
    pairs are used.
    """
    L = 8
    q = 4

    if pairs_from_scratch:
        important_pairs = sample_structures_and_find_pairs(data_utils.get_rna_base_seq(),
                                                           positions, samples=10000) # uncomment to calculate from scratch
    else:
        important_pairs = {(21, 52), (20, 44), (20, 52), (20, 43)}  # pre-calculated

    # add adjacent pairs
    important_pairs = important_pairs.union({(20, 21), (43, 44)})
    V = pairs_to_neighborhoods(self.positions, important_pairs)

    gnk_beta_var = gnk_model.calc_beta_var(L, q, V)
    if return_neighborhoods:
        return gnk_beta_var, V
    else:
        return gnk_beta_var

'''


