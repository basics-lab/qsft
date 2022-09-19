import time

from src.qspright.input_signal_precomputed import PrecomputedSignal
import numpy as np
import itertools
import RNA
from multiprocessing import Pool
from tqdm import tqdm
from src.rna_transform.rna_utils import get_rna_base_seq, _calc_data_inst, insert
from src.qspright.utils import qary_ints, qary_vec_to_dec, save_data
from src.rna_transform.query_iterator import QueryIterator
import sys


class PrecomputedSignalRNA(PrecomputedSignal):
    nucs = np.array(["A", "U", "C", "G"])

    def _init_random(self, **kwargs):
        if self.q != 4:
            raise Exception("For RNA, q must be 4")
        self.base_seq = get_rna_base_seq()
        self.positions = kwargs.get("positions")
        self.parallel = kwargs.get("parallel")
        self.mean = -21.23934478693991
        self.query_args = kwargs.get("query_args")
        self.b = self.query_args.get("b")
        self.num_subsample = self.query_args.get("num_subsample")
        self.num_random_delays = self.query_args.get("num_random_delays")
        self._signal_t = {}
        self._signal_w = {}

    def set_time_domain(self, M, D, save=True, foldername = None, idx = None, save_all_b = False):
        signal_t = {}
        base_inds = []
        freqs = []
        samples = []

        for i in range(self.num_random_delays):
            base_inds.append([((M @ self.L) + np.outer(d, np.ones(self.q ** self.b, dtype=int))) % self.q for d in D[i]])

        iterator = QueryIterator(base_seq=self.base_seq, positions=self.positions, base_inds=base_inds)

        with Pool() as pool:
            y = list(tqdm(pool.imap(_calc_data_inst, iterator), total=len(iterator), miniters=2000))

        start_time = time.time()

        mfes, indices = tuple(zip(*y))
        signal_t = {indices[i]: (mfes[i] - self.mean) for i in range(len(indices))}

        # if idx == 0:
        #     mfes = np.array(mfes).copy()
        #     indices = np.array(indices).copy()
        #     dec_indices = np.array(qary_vec_to_dec(np.array(indices).T, 4)).T.copy()
        #     signal_t_dec = {dec_indices[i]: (mfes[i] - self.mean) for i in range(len(dec_indices))}
        #     signal_t = signal_t.copy()
        #     signal_t_dec = signal_t_dec.copy()
        #
        #     print(dec_indices.shape)
        #
        #     print("size of samples array: ", sys.getsizeof(mfes))
        #
        #     print("size of indices array (q-ary vec): ", sys.getsizeof(indices))
        #     print("size of signal dict (with q-ary keys): ", sys.getsizeof(signal_t))
        #
        #     print("size of indices array (decimal): ", sys.getsizeof(dec_indices))
        #     print("size of signal dict (with decimal keys): ", sys.getsizeof(signal_t_dec))

        end_time = time.time()
        print("Dict creation time: ", end_time - start_time)

        start_time = time.time()

        if save:
            # signal_t_arrays = dict_to_zip(signal_t)
            filename = f"{foldername}/M{idx}.pickle"
            record = (M, D, self.q, signal_t)
            save_data(record, filename)

        end_time = time.time()
        print("File save time: ", end_time - start_time)

        if save_all_b:
            raise Warning("save_all_b is not implemented yet")

        return signal_t