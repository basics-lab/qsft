import time

from src.qspright.input_signal_subsampled import SubsampledSignal
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from src.rna_transform.query_iterator import QueryIterator
from src.qspright.utils import dec_to_qary_vec


class RnaSubsampledSignal(SubsampledSignal):
    nucs = np.array(["A", "U", "C", "G"])

    def __init__(self, **kwargs):
        self.base_seq = kwargs.get("base_seq")
        self.positions = kwargs.get("positions")
        self.sampling_function = kwargs.get("sampling_function")
        self.q = kwargs.get("q")
        if self.q == 4:
            kwargs["n"] = len(self.positions)
        elif self.q == 2:
            kwargs["n"] = 2 * len(self.positions)
        else:
            NotImplementedError(f"q = {self.q} is not valid")
        super().__init__(**kwargs)

    def subsample(self, query_indices):

        batch_size = 10000
        res = []

        #pbar = tqdm(total=len(query_indices), miniters=1000, position=0)
        for i in range(0, len(query_indices), batch_size):
            query_indices_batch_dec = query_indices[i: i+batch_size]
            query_indices_batch = np.array(dec_to_qary_vec(query_indices_batch_dec, self.q, self.n)).T
            with Pool() as pool:
                for new_res in pool.imap(self.sampling_function, query_indices_batch):
                    res.append(new_res)
                    #pbar.update()

        return res