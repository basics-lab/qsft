'''
Class for common interface to an input signal.
'''
from src.qspright.inputsignal import Signal
from src.qspright.inputsignal import random_signal_strength_model
from src.qspright.utils import qary_vec_to_dec, qary_ints
import numpy as np
import random
from src.qspright.utils import fwht, gwht_tensored, igwht_tensored
from src.rna_transform.rna_utils import insert
import RNA
import itertools
from src.rna_transform.rna_utils import _calc_data_inst
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, position=0, leave=True)

class SignalRNA(Signal):

    def __init__(self, **kwargs):
        Signal._init_standard_params(self, **kwargs)
        self.sparsity = kwargs.get("sparsity", 100)
        self.base_seq = kwargs.get("base_seq")
        self.positions = kwargs.get("positions")
        self.parallel = kwargs.get("parallel")
        self.mean = -21.23934478693991
        self.nucs = np.array(["A", "U", "C", "G"])
        self._signal_t = {}

    def set_time_domain(self, Ms, D, logB):
        self.logB = logB
        self.L = np.array(qary_ints(logB, self.q))  # List of all length b qary vectors
        base_inds = []
        for i in range(D.shape[0]):
            base_inds += [((M @ self.L) + np.outer(D[i, :], np.ones(self.q ** self.logB, dtype=int))) % self.q for M in Ms]
        base_inds = np.concatenate(base_inds, axis = -1)
        self.sample(base_inds)

    def sample(self, base_inds):
        if self.parallel:
            sampling_query = []
            for j in tqdm(range(base_inds.shape[1])):
                if tuple(base_inds[:, j]) not in self._signal_t:  # check if sample is already obtained
                    seq = ""
                    for nuc_idx in base_inds[:, j]:
                        seq = seq + self.nucs[nuc_idx]
                    full = insert(self.base_seq, self.positions, seq)
                    sampling_query.append((base_inds[:, j], full))
                    self._signal_t[tuple(base_inds[:, j])] = None

            with Pool() as pool:
                y = list(tqdm(pool.imap(_calc_data_inst, sampling_query), total=len(sampling_query)))

            for sampling_output in y:
                self._signal_t[tuple(sampling_output[0])] = sampling_output[1] - self.mean

        else:
            for j in tqdm(range(base_inds.shape[1])):
                if tuple(base_inds[:, j]) not in self._signal_t: # check if sample is already obtained
                    seq = ""
                    for nuc_idx in base_inds[:, j]:
                        seq = seq + self.nucs[nuc_idx]
                    full = insert(self.base_seq, self.positions, seq)
                    (ss, mfe) = RNA.fold(full)
                    self._signal_t[tuple(base_inds[:, j])] = mfe - self.mean

    def get_time_domain(self, base_inds):
        base_inds = np.array(base_inds)
        if len(base_inds.shape) == 3:
            sample_array = [[tuple(inds[:, i]) for i in range(inds.shape[1])] for inds in base_inds]
            return [np.array([self._signal_t[tup] for tup in inds]) for inds in sample_array]
        elif len(base_inds.shape) == 2:
            sample_array = [tuple(base_inds[:, i]) for i in range(base_inds.shape[1])]
            return np.array([self._signal_t[tup] for tup in sample_array])
