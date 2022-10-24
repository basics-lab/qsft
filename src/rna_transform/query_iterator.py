import numpy as np
from src.qspright.utils import qary_vec_to_dec

class QueryIterator(object):
    nucs = np.array(["A", "U", "C", "G"])
    q = 4

    def __init__(self, base_seq, positions, query_indices):
        self.base_seq = np.array(list(base_seq))
        self.positions = positions
        self.full = self.base_seq.copy()
        self.length = len(query_indices)
        self.query_indices = query_indices
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.length:
            raise StopIteration
        idx = self.query_indices[self.i][0]
        idx_dec = self.query_indices[self.i][1]
        seq = self.nucs[idx]
        self.full[:] = self.base_seq
        self.full[self.positions] = seq
        self.i += 1
        return idx_dec, "".join(self.full)

    def __len__(self):
        return self.length