import time

from src.qspright.input_signal_subsampled import SubsampledSignal
import numpy as np
from multiprocessing import Pool
from src.qspright.utils import dec_to_qary_vec, qary_vec_to_dec
import RNA


class RnaSubsampledSignal(SubsampledSignal):
    nucs = np.array(["A", "U", "C", "G"])

    def __init__(self, **kwargs):
        self.base_seq = kwargs.get("base_seq")
        self.positions = kwargs.get("positions")
        self.sampling_function = kwargs.get("sampling_function")
        self.q = kwargs.get("q")
        self.n = kwargs.get("n")

        (_, mfe_base) = RNA.fold("".join(self.base_seq))
        self.mfe_base = mfe_base

        self.pool = Pool(initializer=worker_init, initargs=(self.base_seq, self.positions, self.mfe_base))

        super().__init__(**kwargs)

        self.pool.close()

    def subsample(self, query_indices):

        batch_size = 250
        res = np.zeros(len(query_indices))
        counter = 0

        # pbar = tqdm(total=len(query_indices), miniters=batch_size, position=0)
        query_batches = np.array_split(query_indices, len(query_indices)//batch_size)

        for new_res in self.pool.imap(sampling_function, query_batches):
            res[counter: counter+len(new_res)] = new_res
            counter += len(new_res)
            # pbar.update(len(new_res))

        return res


def worker_init(base_seq, positions_input, mfe_base_input):
    global base_seq_list
    global positions
    global mfe_base

    base_seq_list = np.array(list(base_seq))
    mfe_base = mfe_base_input
    positions = positions_input


def sampling_function(query_batch):
    global base_seq_list
    global positions
    global mfe_base

    query_batch_q = np.array(dec_to_qary_vec(query_batch, 4, len(positions))).T
    y = []
    for query_index in query_batch_q:
        full = next_q4(base_seq_list, positions, query_index)
        fc = RNA.fold_compound(full)
        (_, mfe) = fc.mfe()
        y.append(mfe - mfe_base)
    return y


nucs = np.array(["A", "U", "C", "G"])

def next_q4(base_seq_list, positions, query_index):
    seq = nucs[query_index]
    base_seq_list[positions] = seq
    return "".join(base_seq_list)


def next_q2(base_seq_list, positions, query_index):
    idx = dec_to_qary_vec([qary_vec_to_dec(query_index, 2)], 4, len(query_index)//2)[:, 0]
    seq = nucs[idx]
    base_seq_list[positions] = seq
    return "".join(base_seq_list)
