'''
Class for common interface to an input signal.
'''
from src.qspright.input_signal import Signal
from src.qspright.utils import qary_vec_to_dec, qary_ints, sort_qary_vecs, random_signal_strength_model
from src.qspright.query import compute_delayed_gwht, get_Ms, get_D, get_Ms_and_Ds
import numpy as np


class LongSignal(Signal):

    def _init_random(self, **kwargs):
        self.query_args = kwargs.get("query_args")
        self.b = self.query_args.get("b")
        self.num_subsample = self.query_args.get("num_subsample")
        self.num_random_delays = self.query_args.get("num_random_delays")
        if self.b is None:
            self.b = np.int(np.maximum(np.log(self.sparsity) / np.log(self.q), 4))
        self.sparsity = kwargs.get("sparsity")
        self.locq = sort_qary_vecs(np.random.randint(self.q, size=(self.n, self.sparsity)).T).T
        self._signal_t = {}
        self._signal_w = {}
        self.a_min = kwargs.get("a_min")
        self.a_max = kwargs.get("a_max")
        self.strengths = random_signal_strength_model(self.sparsity, self.a_min, self.a_max)
        for i in range(self.locq.shape[1]):
            self._signal_w[tuple(self.locq[:, i])] = self.strengths[i]

    def subsample(self):
        self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)
        for D in self.Ds:
            self.set_time_domain(self.Ms, D, self.b)

    def get_MD(self, ret_num_subsample, ret_num_random_delays, b):
        Ms_ret = []
        Ds_ret = []
        if ret_num_subsample <= self.num_subsample and ret_num_random_delays <= self.num_random_delays and b <= self.b:
            subsample_idx = np.random.choice(self.num_subsample, ret_num_subsample, replace=False)
            delay_idx = np.random.choice(self.num_random_delays, ret_num_random_delays, replace=False)
            for i in subsample_idx:
                Ms_ret.append(self.Ms[i][:, (self.b - b):])
                Ds_ret.append([])
                for j in delay_idx:
                    Ds_ret[-1].append(self.Ds[i][j])
            return Ms_ret, Ds_ret

        else:
            raise Exception()

    def _generate_MD(self, **kwargs):
        self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **kwargs)
        for D in self.Ds:
            self.set_time_domain(self.Ms, D, self.b)

    def set_time_domain(self, Ms, D, b, parallel=False):
        self.b = b
        self.Ms = Ms
        self.L = np.array(qary_ints(b, self.q))  # List of all length b qary vectors
        for D_sub in D:
            for i in range(D_sub.shape[0]):
                self.set_time_domain_d(D_sub[i, :])

    def set_time_domain_d(self, d, dec=True):
        base_inds = [((M @ self.L) + np.outer(d, np.ones(self.q ** self.b, dtype=int))) % self.q for M in self.Ms]
        freqs = [k.T @ self.locq for k in base_inds]
        samples = [np.exp(2j*np.pi*freq/self.q) @ self.strengths for freq in freqs]
        for i in range(len(self.Ms)):
            sample = samples[i]
            K = base_inds[i]
            for j in range(self.q ** self.b):
                self._signal_t[tuple(K[:, j])] = sample[j] + self.noise_sd*np.random.normal(loc=0, scale=np.sqrt(2)/2,
                                                                                            size=(1, 2)).view(np.cdouble)

    def get_time_domain(self, base_inds, dec=True):
        base_inds = np.array(base_inds)
        if dec:
            if len(base_inds.shape) == 3:
                sample_array = [qary_vec_to_dec(inds, self.q) for inds in base_inds]
                return [np.array([self._signal_t[tup] for tup in inds]) for inds in sample_array]
            elif len(base_inds.shape) == 2:
                sample_array = [tuple(base_inds[:, i]) for i in range(base_inds.shape[1])]
                return np.array([self._signal_t[tup] for tup in sample_array])
        else:
            if len(base_inds.shape) == 3:
                sample_array = [[tuple(inds[:, i]) for i in range(inds.shape[1])] for inds in base_inds]
                return [np.array([self._signal_t[tup] for tup in inds]) for inds in sample_array]
            elif len(base_inds.shape) == 2:
                sample_array = [tuple(base_inds[:, i]) for i in range(base_inds.shape[1])]
                return np.array([self._signal_t[tup] for tup in sample_array])

    def get_nonzero_locations(self):
        return self.locq.T
