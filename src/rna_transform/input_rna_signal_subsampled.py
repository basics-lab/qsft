import time

from src.qspright.input_signal_subsampled import SubsampledSignal
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from src.rna_transform.query_iterator import QueryIterator


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

        iterator = QueryIterator(base_seq=self.base_seq, positions=self.positions,
                                 query_indices=query_indices, q=self.q)
        # input_query = list(tqdm(iterator))

        with Pool() as pool:
            y = list(tqdm(pool.imap(self.sampling_function, iterator), total=len(iterator),
                          miniters=5000, position=0))

        start_time = time.time()
        signal_t = dict(y)
        end_time = time.time()
        print("Dict creation time: ", end_time - start_time)

        return signal_t