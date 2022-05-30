import numpy as np
import scipy
import itertools
import RNA
from tqdm import tqdm
from sklearn.linear_model import Lasso

import rna_transform.data_utils
import rna_transform.utils
import rna_transform.gnk_model

import sys
sys.path.append("../src")

from qspright.inputsignal import Signal
from qspright.qspright_nso import QSPRIGHT
from qspright.utils import gwht

"""
Utility functions for loading and processing the quasi-empirical RNA fitness function.
"""

RNA_POSITIONS = [2, 3, 4, 20, 21, 30, 43, 44, 52, 70]


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


def load_rna_data(save = False, verbose = False):
    """
    Constructs and returns the data corresponding to the quasi-empirical RNA fitness function
    of the Hammerhead ribozyme HH9. 
    """
    try:
        y = np.load("results/rna_data.npy")
        if verbose:
            print("Loaded saved RNA data.")
        return y - np.mean(y)

    except FileNotFoundError:
        base_seq = get_rna_base_seq()
        base_seq = dna_to_rna(base_seq)
        positions = RNA_POSITIONS
        L = len(positions)
        q = 4

        # construct insertion sequences
        nucs = ["A", "U", "C", "G"]

        y = []
        seqs_as_list = list(itertools.product(nucs, repeat=len(positions)))
        seqs = ["".join(s) for s in seqs_as_list]

        print("Calculating free energies...")
        for s in tqdm(seqs):
            full = insert(base_seq, positions, s)
            (ss, mfe) = RNA.fold(full)
            y.append(mfe)

        y = np.array(y)

        if save:
            np.save("results/rna_data.npy", y)

        return y
    

def generate_householder_matrix():
    nucs = ["A", "U", "C", "G"]
    positions = RNA_POSITIONS

    nucs_idx = {nucs[i]: i for i in range(len(nucs))}
    seqs_as_list = list(itertools.product(nucs, repeat=len(positions)))

    L = len(positions)
    int_seqs = [[nucs_idx[si] for si in s] for s in seqs_as_list]

    print("Constructing Fourier matrix...")
    X = utils.fourier_from_seqs(int_seqs, [4] * L)

    return X


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


def sample_structures_and_find_pairs(base_seq, positions, samples=10000):
    """
    Samples secondary structures from the Boltzmann distribution 
    and finds pairs of positions that are paired in any of the
    sampled strutures.
    """
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(base_seq, md)
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


def calculate_rna_lasso(save=False):
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
        y = load_rna_data(save=save)
        X = generate_householder_matrix()
        print("Fitting Lasso coefficients (this may take some time)...")
        model = Lasso(alpha=alpha, fit_intercept=False)
        model.fit(X, y)
        beta = model.coef_
        if save:
            np.save("results/rna_beta_lasso.npy", beta)
        return beta


def calculate_rna_gwht(save=False):
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
        n = len(RNA_POSITIONS)
        y = load_rna_data(save=save)
        beta = gwht(y, q=4, n=n)
        print("Found GWHT coefficients")
        if save:
            np.save("results/rna_beta_gwht.npy", beta)
        return beta


def calculate_rna_qspright(save=False, report = False, noise_sd=None, verbose = False, num_subsample = 4, num_random_delays = 10, b = None):
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
        y = load_rna_data(save=save, verbose = verbose)
        n = len(RNA_POSITIONS)
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
    V = pairs_to_neighborhoods(RNA_POSITIONS, important_pairs)

    gnk_beta_var = gnk_model.calc_beta_var(L, q, V)
    if return_neighborhoods:
        return gnk_beta_var, V
    else:
        return gnk_beta_var
