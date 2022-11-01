import random
import numpy as np

from src.qspright.utils import igwht_tensored, random_signal_strength_model, qary_vec_to_dec, sort_qary_vecs
from src.qspright.input_signal import Signal
from src.qspright.input_signal_subsampled import SubsampledSignal


def generate_signal_w(n, q, noise_sd, sparsity, a_min, a_max, full=True, max_weight=None):
    max_weight = n if max_weight is None else max_weight
    N = q ** n
    if max_weight == n:
        locq = sort_qary_vecs(np.random.randint(q, size=(n, sparsity)).T).T
    else:
        non_zero_idx_vals = np.random.randint(q-1, size=(max_weight, sparsity))+1
        non_zero_idx_pos = np.random.choice(a=n, size=(sparsity, max_weight))
        locq = np.zeros((n, sparsity), dtype=int)
        for i in range(sparsity):
            locq[non_zero_idx_pos[i, :], i] = non_zero_idx_vals[:, i]
        locq = sort_qary_vecs(locq.T).T
    loc = qary_vec_to_dec(locq, q)
    strengths = random_signal_strength_model(sparsity, a_min, a_max)
    if full:
        wht = np.zeros((N,), dtype=complex)
        for l, s in zip(loc, strengths):
            wht[l] = s
        signal_w = wht + np.random.normal(0, noise_sd, size=(N, 2)).view(np.complex).reshape(N)
        return np.reshape(signal_w, [q] * n), locq, strengths
    else:
        signal_w = dict(zip(list(map(tuple, locq.T)), strengths))
        return signal_w, locq, strengths


def get_random_signal(n, q, noise_sd, sparsity, a_min, a_max):
    signal_w, locq, strengths = generate_signal_w(n, q, noise_sd, sparsity, a_min, a_max, full=True)
    signal_t = igwht_tensored(signal_w, q, n)
    signal_params = {
        "n": n,
        "q": q,
        "noise_sd": noise_sd,
        "signal_t": signal_t,
        "signal_w": signal_w,
        "folder": "test_data"
    }
    return SyntheticSignal(locq, strengths, **signal_params)


class SyntheticSignal(Signal):

    def __init__(self, locq, strengths, **kwargs):
        super().__init__(**kwargs)
        self.locq = locq
        self.strengths = strengths


def get_random_subsampled_signal(n, q, noise_sd, sparsity, a_min, a_max, query_args, max_weight=None):
    signal_w, locq, strengths = generate_signal_w(n, q, noise_sd, sparsity, a_min, a_max, full=False, max_weight=max_weight)
    signal_params = {
        "n": n,
        "q": q,
        "query_args": query_args,
    }
    return SyntheticSubsampledSignal(signal_w=signal_w, locq=locq, strengths=strengths,
                                     noise_sd=noise_sd, **signal_params)


class SyntheticSubsampledSignal(SubsampledSignal):

    def __init__(self, signal_w, locq, strengths, noise_sd, **kwargs):
        self.locq = locq
        self.strengths = strengths
        self.noise_sd = noise_sd
        super().__init__(**kwargs, signal_w=signal_w)

    def subsample(self, query_indices):
        signal_t = {}
        N = len(query_indices)
        k = np.array([query_indices[i][0] for i in range(N)]).T
        freq = (k.T @ self.locq) % self.q
        samples = np.exp(2j * np.pi * freq / self.q) @ self.strengths
        for i in range(N):
            signal_t[query_indices[i][1]] = \
                np.csingle(samples[i] + self.noise_sd * np.random.normal(loc=0, scale=np.sqrt(2) / 2,
                                                                               size=(1, 2)).view(np.cdouble))
        return signal_t
