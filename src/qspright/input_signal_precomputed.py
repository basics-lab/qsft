from src.qspright.input_signal_long import LongSignal
from query import get_Ms_and_Ds
from utils import qary_ints
import numpy as np
import pickle
from pathlib import Path

class PrecomputedSignal(LongSignal):

    def __init__(self, **kwargs):
        if kwargs.get("signal") is None:
            self._init_standard_params(**kwargs)
            self._init_random(**kwargs)
            self.L = np.array(qary_ints(self.b, self.q))  # List of all length b qary vectors
        else:
            self._init_given(**kwargs)

    def _init_given(self, **kwargs):
        self.noise_sd = kwargs.get("noise_sd")
        if kwargs.get("M_select"):
            self._init_given_foldertype(**kwargs)
        else:
            self._init_given_filetype(**kwargs)
        if kwargs.get("transform"):
            with open(kwargs.get("transform"), 'rb') as f:
                self._signal_w, self.locq = pickle.load(f)
            f.close()
    def _init_given_filetype(self, **kwargs):
        filename = kwargs.get("signal")
        with open(filename, 'rb') as f:
            self.Ms, self.Ds, self.q, self._signal_t = pickle.load(f)
        f.close()
        self.n, self.b = self.Ms[0].shape
        self.num_subsample = len(self.Ms)
        self.num_random_delays = len(self.Ds[0])

    def _init_given_foldertype(self, **kwargs):
        foldername = kwargs.get("signal")
        M_select = kwargs.get("M_select")
        b = kwargs.get("b")
        self.Ms = []
        self.Ds = []
        self._signal_t = {}
        self.num_subsample = 0
        for i in range(len(M_select)):
            if M_select[i]:
                filename = f"{foldername}/M{i}.pickle" if b is None else f"{foldername}/M{i}_b{b}.pickle"
                with open(filename, 'rb') as f:
                    M, D, self.q, signal_t = pickle.load(f)
                f.close()
                self.n, self.b = M.shape
                self.num_subsample += 1
                self.num_random_delays = len(D)
                self.Ms.append(M)
                self.Ds.append(D)
                self._signal_t |= signal_t


    def subsample(self, foldername, all_b=False, save_locally=False):
        self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)
        Path(f"./{foldername}").mkdir(exist_ok=True)
        for (M, D, i) in zip(self.Ms, self.Ds, range(self.num_subsample)):
            if save_locally:
                self._signal_t |= self.set_time_domain(M, D, foldername, i, all_b)
            else:
                self.set_time_domain(M, D, foldername, i, all_b)

    def subsample_nosave(self):
        super.subsample(self)
    def set_time_domain(self, M, D, foldername, idx, all_b):
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
                    if i == 0 and j == 0 and all_b and r == (self.q ** b_i):
                        filename = f"{foldername}/M{idx}_b{b_i}.pickle"
                        with open(filename, 'wb') as f:
                            pickle.dump((M[:, (self.b - b_i):], D, self.q, signal_t), f)
                        f.close()
                        b_i += 1
                    signal_t[tuple(base_inds[i][j][:, r])] = np.csingle(samples[i][j][r] + self.noise_sd*np.random.normal(loc=0, scale=np.sqrt(2)/2,
                                                                                        size=(1, 2)).view(np.cdouble))
        filename = f"{foldername}/M{idx}_b{b_i}.pickle" if all_b else f"{foldername}/M{idx}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump((M, D, self.q, signal_t), f)
        f.close()
        return signal_t

    def save_full_signal(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.Ms, self.Ds, self.q, self._signal_t), f)
        f.close()

    def save_transform(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._signal_w, self.locq), f)

