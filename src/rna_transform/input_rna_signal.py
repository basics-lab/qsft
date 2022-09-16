from src.qspright.inputsignal import Signal
import numpy as np
import RNA
from src.qspright.utils import qary_vec_to_dec
import itertools
from src.rna_transform.rna_utils import get_rna_base_seq, _calc_data_inst, insert
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, position=0, leave=True)

class SignalRNA(Signal):
    nucs = np.array(["A", "U", "C", "G"])

    def __init__(self, **kwargs):
        self._init_standard_params(**kwargs)
        self.base_seq = get_rna_base_seq()
        self.positions = kwargs.get("positions")
        self.parallel = kwargs.get("parallel")
        self.mean = -21.23934478693991

    def sample(self):
        nucs = ["A", "U", "C", "G"]
        seqs_as_list = list(itertools.product(nucs, repeat=len(self.positions)))
        seqs = np.array(["".join(s) for s in seqs_as_list])

        if self.parallel:

            query = []
            for i, s in enumerate(seqs):
                full = insert(self.base_seq, self.positions, s)
                query.append(full)

            with Pool() as pool:
                y = list(tqdm(pool.imap(_calc_data_inst, query), total=len(seqs)))

            self._signal_t = np.array(y)

        else:
            y = []
            for s in tqdm(seqs):
                full = insert(self.base_seq, self.positions, s)
                (ss, mfe) = RNA.fold(full)
                y.append(mfe - self.mean)
            self._signal_t = np.array(y)

    def get_time_domain(self, base_inds):
        base_inds = np.array(base_inds)
        if len(base_inds.shape) == 3:
            sample_array = [[tuple(inds[:, i]) for i in range(inds.shape[1])] for inds in base_inds]
            return [np.array(self._signal_t[inds]) for inds in sample_array]
        elif len(base_inds.shape) == 2:
            sample_array = [tuple(base_inds[:, i]) for i in range(base_inds.shape[1])]
            return np.array(self._signal_t[sample_array])

