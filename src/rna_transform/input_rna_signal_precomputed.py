from src.qspright.input_signal_precomputed import PrecomputedSignal
import numpy as np
import itertools
import RNA
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from src.rna_transform.rna_utils import  insert, dna_to_rna, get_rna_base_seq, _calc_data_inst


class PrecomputedSignalRNA(PrecomputedSignal):
    nucs = np.array(["A", "U", "C", "G"])

    def _init_random(self, **kwargs):
        if self.q != 4:
            raise Exception("For RNA, q must be 4")
        self.sparsity = kwargs.get("sparsity", 100)
        self.base_seq = dna_to_rna(get_rna_base_seq())
        self.positions = kwargs.get("positions")
        self.parallel = kwargs.get("parallel")
        self.mean = -21.23934478693991
        self.query_args = kwargs.get("query_args")
        self.b = self.query_args.get("b")
        self.num_subsample = self.query_args.get("num_subsample")
        self.num_random_delays = self.query_args.get("num_random_delays")
        nucs = ["A", "U", "C", "G"]
        seqs_as_list = list(itertools.product(nucs, repeat=len(self.positions)))
        self.seqs = np.array(["".join(s) for s in seqs_as_list])
        self._signal_t = {}
        self._signal_w = {}

    def set_time_domain(self, M, D, foldername, idx, all_b):
        signal_t = {}
        base_inds = []
        freqs = []
        samples = []
        b_min = 2
        b_i = b_min
        for i in range(self.num_random_delays):
            base_inds.append([((M @ self.L) + np.outer(d, np.ones(self.q ** self.b, dtype=int))) % self.q for d in D[i]])
        for r in range(self.q ** self.b):
            for i in range(self.num_random_delays):
                for j in range(len(D[0])):
                    if i == 0 and j == 0 and all_b and r == (self.q ** b_i):
                        filename = f"{foldername}/M{idx}_b{b_i}.pickle"
                        with open(filename, 'wb') as f:
                            pickle.dump((M[:, (self.b - b_i):], D, self.q, signal_t), f)
                        f.close()
                        b_i += 1
                    seq = ""
                    for nuc_idx in base_inds[i][j][:, r]:
                        seq = seq + self.nucs[nuc_idx]
                    full = insert(self.base_seq, self.positions, seq)
                    (ss, mfe) = RNA.fold(full)
                    signal_t[tuple(base_inds[i][j][:, r])] = np.csingle(mfe - self.mean)
        filename = f"{foldername}/M{idx}_b{b_i}.pickle" if all_b else f"{foldername}/M{idx}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump((M, D, self.q, signal_t), f)
        f.close()
        return signal_t