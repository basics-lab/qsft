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
        foldername = kwargs.get("signal")
        M_select = kwargs.get("M_select")
        self.Ms = []
        self.Ds = []
        self._signal_t = {}
        self.num_subsample = 0
        for i in range(len(M_select)):
            if M_select[i]:
                with open(f"{foldername}/M{i}.pickle", 'rb') as f:
                    M, D, self.q, signal_t = pickle.load(f)
                f.close()
                self.n, self.b = M.shape
                self.num_subsample += 1
                self.num_random_delays = len(D)
                self.Ms.append(M)
                self.Ds.append(D)
                self._signal_t |= signal_t
        breakpoint()

        f.close()
        if kwargs.get("transform"):
            with open(kwargs.get("transform"), 'rb') as f:
                self._signal_w, self.locq = pickle.load(f)

    def subsample(self, foldername):
        self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)
        Path(f"./{foldername}").mkdir(exist_ok=True)
        for (M, D, i) in zip(self.Ms, self.Ds, range(self.num_subsample)):
            self.set_time_domain(M, D, foldername, i)

    def set_time_domain(self, M, D, foldername, idx):
        signal_t = {}
        base_inds = []
        freqs = []
        samples = []
        for i in range(self.num_random_delays):
            base_inds.append([((M @ self.L) + np.outer(d, np.ones(self.q ** self.b, dtype=int))) % self.q for d in D[i]])
            freqs.append([k.T @ self.locq for k in base_inds[i]])
            samples.append([np.exp(2j * np.pi * freq / self.q) @ self.strengths for freq in freqs[i]])
        for r in range(self.q ** self.b):
            for i in range(self.num_random_delays):
                for j in range(len(D[0])):
                    signal_t[tuple(base_inds[i][j][:, r])] = samples[i][j][r] + self.noise_sd*np.random.normal(loc=0, scale=np.sqrt(2)/2,
                                                                                            size=(1, 2)).view(np.cdouble)
        filename = f"{foldername}/M{idx}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump((M, D, self.q, signal_t), f)
        f.close()


    def save_signal(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.Ms, self.Ds, self.q, self._signal_t), f)
        f.close()

    def save_transform(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._signal_w, self.locq), f)

