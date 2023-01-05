from qspright.input_signal import Signal
import numpy as np
import itertools
from src.rna_transform.rna_utils import insert
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, position=0, leave=True)

class RnaSignal(Signal):
    def __init__(self, **kwargs):
        self.base_seq = kwargs.get("base_seq")
        self.positions = kwargs.get("positions")
        self.n = len(self.positions)
        self.sampling_function = kwargs.get("sampling_function")
        super().__init__(**kwargs)

    def sample(self):
        nucs = ["A", "U", "C", "G"]
        seqs_as_list = list(itertools.product(nucs, repeat=len(self.positions)))
        seqs = np.array(["".join(s) for s in seqs_as_list])

        query = []
        for i, s in enumerate(seqs):
            full = insert(self.base_seq, self.positions, s)
            query.append(full)

        raise NotImplementedError("things need to be changed")

        with Pool() as pool:
            y = list(tqdm(pool.imap(self.sampling_function, query), total=len(seqs)))

        self._signal_t = np.array(y)