'''
Class for common interface to an input signal.
'''
from src.qspright.inputsignal import Signal
from src.qspright.inputsignal import random_signal_strength_model
from src.qspright.utils import qary_vec_to_dec, qary_ints
import numpy as np
import random
from src.qspright.utils import fwht, gwht_tensored, igwht_tensored

class LongSignal(Signal):

    def _init_random(self, **kwargs):
        self.sparsity = kwargs.get("sparsity")
        self.locq = np.random.randint(self.q, size=(self.n, self.sparsity))
        self._signal_t = {}
        self._signal_w = {}
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.strengths = random_signal_strength_model(self.sparsity, self.a, self.b)
        for i in range(self.locq.shape[1]):
            self._signal_w[tuple(self.locq[:, i])] = self.strengths[i]

    def set_time_domain(self, Ms, D, logB):
        self.logB = logB
        self.L = np.array(qary_ints(logB, self.q))  # List of all length b qary vectors
        for i in range(D.shape[0]):
            self.set_time_domain_d(Ms, D[i, :])

    def set_time_domain_d(self, Ms, d):
        base_inds = [((M @ self.L) + np.outer(d, np.ones(self.q ** self.logB, dtype=int))) % self.q for M in Ms]
        freqs = [k.T @ self.locq for k in base_inds]
        samples = [np.exp(2j*np.pi*freq/self.q) @ self.strengths for freq in freqs]
        for i in range(len(Ms)):
            sample = samples[i]
            K = base_inds[i]
            for j in range(self.q ** self.logB):
                self._signal_t[tuple(K[:, j])] = sample[j]

    def get_time_domain(self, base_inds):
        base_inds = np.array(base_inds)
        if len(base_inds.shape) == 3:
            sample_array = [[tuple(inds[:, i]) for i in range(self.q ** self.logB)] for inds in base_inds]
            return [np.array([self._signal_t[tup] for tup in inds]) for inds in sample_array]
        elif len(base_inds.shape) == 2:
            sample_array = [tuple(base_inds[:, i]) for i in range(self.q ** self.logB)]
            return np.array([self._signal_t[tup] for tup in sample_array])

    def get_nonzero_locations(self):
        for i in range(self.sparsity):
            return qary_vec_to_dec(self.locq, self.q)
