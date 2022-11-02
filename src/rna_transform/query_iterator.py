import numpy as np
from src.qspright.utils import dec_to_qary_vec

class QueryIterator(object):
    nucs = np.array(["A", "U", "C", "G"])
    q = 4

    def __init__(self, base_seq, positions, query_indices, q):
        self.base_seq = np.array(list(base_seq))
        self.positions = positions
        self.full = self.base_seq.copy()
        self.length = len(query_indices)
        self.q = q
        self.query_indices = query_indices
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.q == 4:
            return self._next_q4()
        elif self.q == 2:
            return self._next_q2()

    def _next_q4(self):
        if self.i == self.length:
            raise StopIteration
        idx = self.query_indices[self.i][0]
        idx_dec = self.query_indices[self.i][1]
        seq = self.nucs[idx]
        self.full[:] = self.base_seq
        self.full[self.positions] = seq
        self.i += 1
        return idx_dec, "".join(self.full)

    def _next_q2(self):
        if self.i == self.length:
            raise StopIteration
        idx_bin = self.query_indices[self.i][0]
        idx_dec = self.query_indices[self.i][1]
        idx = dec_to_qary_vec([idx_dec], 4, len(idx_bin)//2)[:, 0]
        seq = self.nucs[idx]
        self.full[:] = self.base_seq
        self.full[self.positions] = seq
        self.i += 1
        return idx_dec, "".join(self.full)

    def __len__(self):
        return self.length