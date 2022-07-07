from qspright.inputsignal import Signal
from rna_transform.rna_utils import insert
import numpy as np
import RNA


class SignalRNA(Signal):
    nucs = np.array(["A", "U", "C", "G"])

    def __init__(self, **kwargs):
        Signal._init_standard_params(self, **kwargs)
        self.sparsity = kwargs.get("sparsity", 100)
        self.base_seq = kwargs.get("base_seq")
        self.positions = kwargs.get("positions")

    def get_time_domain(self, inds):
        y = []
        for idx in inds:
            s = self.nucs[idx]
            full = insert(self.base_seq, self.positions, s)
            (ss, mfe) = RNA.fold(full)
            y.append(mfe)
        return y