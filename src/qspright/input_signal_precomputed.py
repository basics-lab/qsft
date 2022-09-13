'''
Class for common interface to an input signal.
'''
from src.qspright.inputsignal import Signal
from src.qspright.inputsignal import random_signal_strength_model
from src.qspright.utils import qary_vec_to_dec, qary_ints
from src.qspright.query import compute_delayed_gwht, get_Ms, get_D

import time
import numpy as np
from multiprocessing import Pool
import random
from src.qspright.utils import fwht, gwht_tensored, igwht_tensored


class PrecomputedSignal(Signal):

    def _init_random(self, **kwargs):
        self.sparsity = kwargs.get("sparsity")
        self.locq = np.random.randint(self.q, size=(self.n, self.sparsity))
        self._signal_t = {}
        self._signal_w = {}
        self.a_min = kwargs.get("a_min")
        self.a_max = kwargs.get("a_max")
        self.strengths = random_signal_strength_model(self.sparsity, self.a_min, self.a_max)
        for i in range(self.locq.shape[1]):
            self._signal_w[tuple(self.locq[:, i])] = self.strengths[i]

    def subsample(self, **kwargs):
        self.query_method = kwargs.get("query_method")
        self.delays_method = kwargs.get("delays_method")
        self.num_subsample = kwargs.get("num_subsample")
        self.num_random_delays = kwargs.get("num_random_delays")
        self.b = kwargs.get("b")

        if self.b is None:
            b = np.int(np.maximum(np.log(signal.sparsity) / np.log(self.q), 4))
        else:
            b = self.b

        self._generate_MD(**kwargs)

    def get_MDU(self, ret_num_subsample, ret_num_random_delays):
        Ms_ret = []
        Ds_ret = []
        Us_ret = []
        used_ret = []

        if ret_num_subsample <= self.num_subsample and ret_num_random_delays <= self.num_random_delays:
            subsample_idx = np.random.choice(self.num_subsample, ret_num_subsample, replace=False)
            delay_idx = np.random.choice(self.num_random_delays, ret_num_random_delays, replace=False)

            for i in subsample_idx:
                Ms_ret.append(self.Ms[i])
                Ds_ret.append([])
                Us_ret.append([])
                used_ret.append([])
                for j in delay_idx:
                    Ds_ret[-1].append(self.Ds[i][j])
                    Us_ret[-1].append(self.Us[i][j])
                    used_ret[-1].append(self.used[i][j])

            return Ms_ret, Ds_ret, Us_ret, used_ret

        else:
            raise Exception()


    def _generate_MD(self, **kwargs):

        timing_verbose = kwargs.get("timing_verbose", True)
        verbose = kwargs.get("verbose", False)
        report = kwargs.get("report", True)

        if timing_verbose:
            start_time = time.time()

        self.Ms = get_Ms(self.n, self.b, self.q, method=self.query_method, num_to_get=self.num_subsample)

        if timing_verbose:
            print(f"M Generation:{time.time() - start_time}")

        self.Us = []
        self.Ds = []

        if self.delays_method == "identity":
            self.num_delays = self.n + 1
        elif self.delays_method == "nso":
            self.num_delays = self.num_random_delays * (self.n + 1)
        else:
            self.num_delays = self.num_random_delays

        if timing_verbose:
            start_time = time.time()

        D = get_D(self.n, method=self.delays_method, num_delays=self.num_delays, q=self.q)

        if timing_verbose:
            print(f"D Generation:{time.time() - start_time}")
            start_time = time.time()

        self.set_time_domain(self.Ms, D, self.b)

        if timing_verbose:
            print(f"Signal Sampling:{time.time() - start_time}")
            start_time = time.time()

        self.used = []

        # subsample with shifts [D], make the observation [U]
        for M in self.Ms:
            if verbose:
                print("------")
                print("subsampling matrix")
                print(M)
                print("delay matrix")
                print(D)
            self.Ds.append(D)
            U = []
            used = []
            for D_sub in D:
                U_sub, used_i = compute_delayed_gwht(self, M, D_sub, self.q)
                U.append(U_sub)
                used.append(used_i)
            self.Us.append(U)
            self.used.append(used)

        if timing_verbose:
            print(f"Fourier Transformation Total Time:{time.time() - start_time}")

    def set_time_domain(self, Ms, D, b, parallel=False):
        self.b = b
        self.Ms = Ms
        self.L = np.array(qary_ints(b, self.q))  # List of all length b qary vectors
        for D_sub in D:
            for i in range(D_sub.shape[0]):
                self.set_time_domain_d(D_sub[i, :])

    def set_time_domain_d(self, d):
        base_inds = [((M @ self.L) + np.outer(d, np.ones(self.q ** self.b, dtype=int))) % self.q for M in self.Ms]
        freqs = [k.T @ self.locq for k in base_inds]
        samples = [np.exp(2j*np.pi*freq/self.q) @ self.strengths for freq in freqs]
        for i in range(len(self.Ms)):
            sample = samples[i]
            K = base_inds[i]
            for j in range(self.q ** self.b):
                self._signal_t[tuple(K[:, j])] = sample[j] + self.noise_sd*np.random.normal(loc=0, scale=np.sqrt(2)/2,
                                                                                            size=(1, 2)).view(np.cdouble)

    def get_time_domain(self, base_inds):
        base_inds = np.array(base_inds)
        if len(base_inds.shape) == 3:
            sample_array = [[tuple(inds[:, i]) for i in range(self.q ** self.b)] for inds in base_inds]
            return [np.array([self._signal_t[tup] for tup in inds]) for inds in sample_array]
        elif len(base_inds.shape) == 2:
            sample_array = [tuple(base_inds[:, i]) for i in range(self.q ** self.b)]
            return np.array([self._signal_t[tup] for tup in sample_array])

    def get_nonzero_locations(self):
        return qary_vec_to_dec(self.locq, self.q)
