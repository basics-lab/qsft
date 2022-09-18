from src.qspright.input_signal_long import LongSignal
from src.qspright.query import get_Ms_and_Ds
from src.qspright.utils import qary_ints, zip_to_dict, dict_to_zip
import numpy as np
import pickle
from pathlib import Path
import time

class PrecomputedSignal(LongSignal):

    def __init__(self, **kwargs):
        print(kwargs)
        if kwargs.get("signal") is None:
            self._init_standard_params(**kwargs)
            self._init_random(**kwargs)
            self.L = np.array(qary_ints(self.b, self.q))  # List of all length b qary vectors
        else:
            self._init_given(**kwargs)

    def _init_given(self, **kwargs):
        self.noise_sd = kwargs.get("noise_sd")
        start_time = time.time()

        if kwargs.get("M_select"):
            self._init_given_foldertype(**kwargs)
        else:
            self._init_given_filetype(**kwargs)
        if kwargs.get("transform"):
            with open(kwargs.get("transform"), 'rb') as f:
                self._signal_w, self.locq = pickle.load(f)

        end_time = time.time()
        print("Data load: ", end_time - start_time)

    def _init_given_filetype(self, **kwargs):
        filename = kwargs.get("signal")
        with open(filename, 'rb') as f:
            self.Ms, self.Ds, self.q, signal_t_arrays = pickle.load(f)

        self.n, self.b = self.Ms[0].shape
        self._signal_t = zip_to_dict(signal_t_arrays, self.n)
        self.num_subsample = len(self.Ms)
        self.num_random_delays = len(self.Ds[0])

    def _init_given_foldertype(self, **kwargs):
        foldername = kwargs.get("signal")
        M_select = kwargs.get("M_select")
        b = kwargs.get("b")

        # check if all files exist
        for i in range(len(M_select)):
            if M_select[i]:
                file = Path(f"{foldername}/M{i}.pickle" if b is None else f"{foldername}/M{i}_b{b}.pickle")
                if not file.is_file():
                    raise FileNotFoundError

        self.Ms = []
        self.Ds = []
        self._signal_t = {}
        self.num_subsample = 0
        for i in range(len(M_select)):
            if M_select[i]:
                filename = f"{foldername}/M{i}.pickle" if b is None else f"{foldername}/M{i}_b{b}.pickle"
                with open(filename, 'rb') as f:
                    M, D, self.q, signal_t_arrays = pickle.load(f)

                self.n, self.b = M.shape
                signal_t = zip_to_dict(signal_t_arrays, self.n)
                self.num_subsample += 1
                self.num_random_delays = len(D)
                self.Ms.append(M)
                self.Ds.append(D)
                self._signal_t.update(signal_t)


    def subsample(self, keep_samples=True, save_samples_to_file = False, foldername = None, save_all_b=False):
        self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)
        Path(f"./{foldername}").mkdir(exist_ok=True)
        for (M, D, i) in zip(self.Ms, self.Ds, range(self.num_subsample)):
            if keep_samples:
                self._signal_t.update(self.set_time_domain(M, D, save_samples_to_file, foldername, i, save_all_b))
            else:
                self.set_time_domain(M, D, save_samples_to_file, foldername, i, save_all_b)

    def subsample_nosave(self):
        super.subsample(self)

    def set_time_domain(self, M, D, save, foldername = None, idx = None, save_all_b = None):
        signal_t = {}
        base_inds = []
        freqs = []
        samples = []
        b_min = 2
        b_i = b_min
        for i in range(self.num_random_delays):
            base_inds.append([((M @ self.L) + np.outer(d, np.ones(self.q ** self.b, dtype=int))) % self.q for d in D[i]])
            freqs.append([k.T @ self.locq for k in base_inds[i]])
            samples.append([np.exp(2j * np.pi * freq / self.q) @ self.strengths for freq in freqs[i]])
        for r in range(self.q ** self.b):
            for i in range(self.num_random_delays):
                for j in range(len(D[0])):
                    if i == 0 and j == 0 and save and save_all_b and r == (self.q ** b_i):
                        filename = f"{foldername}/M{idx}_b{b_i}.pickle"
                        with open(filename, 'wb') as f:
                            signal_t_arrays = dict_to_zip(signal_t)
                            pickle.dump((M[:, (self.b - b_i):], D, self.q, signal_t_arrays), f)
                        b_i += 1
                    signal_t[tuple(base_inds[i][j][:, r])] = np.csingle(samples[i][j][r] + self.noise_sd*np.random.normal(loc=0, scale=np.sqrt(2)/2,
                                                                                        size=(1, 2)).view(np.cdouble))
        if save:
            filename = f"{foldername}/M{idx}_b{b_i}.pickle" if save_all_b else f"{foldername}/M{idx}.pickle"
            with open(filename, 'wb') as f:
                signal_t_arrays = dict_to_zip(signal_t)
                pickle.dump((M, D, self.q, signal_t_arrays), f)

        return signal_t

    def save_full_signal(self, filename):
        with open(filename, 'wb') as f:
            signal_t_arrays = dict_to_zip(self._signal_t)
            pickle.dump((self.Ms, self.Ds, self.q, signal_t_arrays), f)
        f.close()

    def save_transform(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._signal_w, self.locq), f)

