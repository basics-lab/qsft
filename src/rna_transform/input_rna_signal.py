from qspright.inputsignal import Signal
from rna_transform.rna_utils import insert
import numpy as np
import RNA
from qspright.utils import qary_vec_to_dec
import itertools
from rna_transform.rna_utils import _calc_data_inst
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, position=0, leave=True)

class SignalRNA(Signal):
    nucs = np.array(["A", "U", "C", "G"])

    def __init__(self, **kwargs):
        Signal._init_standard_params(self, **kwargs)
        self.sparsity = kwargs.get("sparsity", 100)
        self.base_seq = kwargs.get("base_seq")
        self.positions = kwargs.get("positions")
        self.parallel = kwargs.get("parallel")
        self.mean = -21.23934478693991

        nucs = ["A", "U", "C", "G"]
        seqs_as_list = list(itertools.product(nucs, repeat=len(self.positions)))
        self.seqs = np.array(["".join(s) for s in seqs_as_list])

    def get_time_domain(self, inds):

        inds = np.array(inds)
        if len(inds.shape) == 3:
            multidimensional = True
        else:
            multidimensional = False

        if multidimensional:
            inds = np.moveaxis(inds, [0,1,2], [1,0,2])
            original_shape = inds.shape
            inds = inds.reshape((original_shape[0], -1))

        inds_dec = np.array(qary_vec_to_dec(np.array(inds), self.q), dtype = np.int32)

        # print("calculating for ", len(inds_dec), " samples")

        if self.parallel:

            sub_seqs = self.seqs[inds_dec]
            with Pool() as pool:
                y = list(tqdm(pool.imap(_calc_data_inst, zip(itertools.repeat(self.base_seq),
                                                             itertools.repeat(self.positions), sub_seqs)),
                              total=len(sub_seqs)))

        else:
            y = []
            for idx in inds_dec:
                s = self.seqs[idx]
                full = insert(self.base_seq, self.positions, s)
                (ss, mfe) = RNA.fold(full)
                y.append(mfe)

        y = np.array(y) - self.mean

        if multidimensional:
            y = y.reshape(original_shape[1:])

        return y

