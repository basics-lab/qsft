import numpy as np
from src.rna_transform.rna_utils import insert
from src.qspright.utils import qary_vec_to_dec

class QueryIterator(object):
    nucs = np.array(["A", "U", "C", "G"])
    q = 4

    def __init__(self, base_seq, positions, base_inds):
        self.base_seq = base_seq
        self.positions = positions
        self.num_random_delays = len(base_inds)
        self.num_subdelays = len(base_inds[0])
        self.q_pow_b = base_inds[0][0].shape[1]
        self.base_inds = base_inds
        self.base_inds_dec = []
        for i in range(self.num_random_delays):
            base_inds_dec_sub = []
            for j in range(self.num_subdelays):
                inds_ij = base_inds[i][j]
                base_inds_dec_sub.append(qary_vec_to_dec(inds_ij, self.q))
            self.base_inds_dec.append(base_inds_dec_sub)
        self.r = 0
        self.i = 0
        self.j = 0
        self.base_inds_current = self.base_inds[self.i][self.j]
        self.base_inds_dec_current = self.base_inds_dec[self.i][self.j]

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.num_random_delays:
            raise StopIteration
        idx = self.base_inds_current[:, self.r]
        idx_dec = self.base_inds_dec_current[self.r]
        seq = self.nucs[idx]
        full = insert(self.base_seq, self.positions, seq)
        self.update_counts()
        return (full, idx_dec)

    def update_counts(self):
        self.r += 1
        if self.r == self.q_pow_b:
            self.r = 0
            self.j += 1
            if self.j == self.num_subdelays:
                self.j = 0
                self.i += 1
            if not self.i == self.num_random_delays:
                self.base_inds_current = self.base_inds[self.i][self.j]
                self.base_inds_dec_current = self.base_inds_dec[self.i][self.j]

    def __len__(self):
        return self.q_pow_b * self.num_random_delays * self.num_subdelays