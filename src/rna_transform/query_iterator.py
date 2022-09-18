import numpy as np
from src.rna_transform.rna_utils import insert

class QueryIterator(object):
    nucs = np.array(["A", "U", "C", "G"])

    def __init__(self, base_seq, positions, base_inds):
        self.base_seq = base_seq
        self.positions = positions
        self.num_random_delays = len(base_inds)
        self.num_subdelays = len(base_inds[0])
        self.q_pow_b = base_inds[0][0].shape[1]
        self.base_inds = base_inds
        self.r = 0
        self.i = 0
        self.j = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.r == self.q_pow_b:
            raise StopIteration
        seq = self.nucs[self.base_inds[self.i][self.j][:, self.r]]
        full = insert(self.base_seq, self.positions, seq)
        self.update_counts()
        return full

    def update_counts(self):
        self.j += 1
        if self.j == self.num_subdelays:
            self.j = 0
            self.i += 1
            if self.i == self.num_random_delays:
                self.i = 0
                self.r += 1

    def __len__(self):
        return self.q_pow_b * self.num_random_delays * self.num_subdelays