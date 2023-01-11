import numpy as np
from qsft.utils import igwht_tensored, random_signal_strength_model, qary_vec_to_dec, sort_qary_vecs
from qsft.input_signal import Signal
from qsft.input_signal_subsampled import SubsampledSignal
from qsft.utils import dec_to_qary_vec
from multiprocess import Pool
import time


def generate_signal_w(n, q, sparsity, a_min, a_max, noise_sd=0, full=True, max_weight=None):
    """
    Generates a sparse fourier transform
    """
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
    """
    Computes a full random time-domain signal, which is sparse in the fequency domain. This function is only suitable for
    small n since for large n, storing all q^n symbols is not tractable.
    """
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
    """
    This is essentially just a signal object, except the strengths and locations of the non-zero indicies are known, and
    included as attributes
    """
    def __init__(self, locq, strengths, **kwargs):
        super().__init__(**kwargs)
        self.locq = locq
        self.strengths = strengths


def get_random_subsampled_signal(n, q, noise_sd, sparsity, a_min, a_max, query_args, max_weight=None):
    """
    Similar to get_random_signal, but instead of returning a SyntheticSignal object, it returns a SyntheticSubsampledSignal
    object. The advantage of this is that a subsampled signal does not compute the time domain signal on creation, but
    instead, creates it on the fly. This should be used (1) when n is large or (2) when sampling is expensive.
    """
    start_time = time.time()
    signal_w, locq, strengths = generate_signal_w(n, q, sparsity, a_min, a_max, noise_sd, full=False, max_weight=max_weight)
    signal_params = {
        "n": n,
        "q": q,
        "query_args": query_args,
    }
    print(f"Generation Time:{time.time() - start_time}", flush=True)
    return SyntheticSubsampledSignal(signal_w=signal_w, locq=locq, strengths=strengths,
                                     noise_sd=noise_sd, **signal_params)


class SyntheticSubsampledSignal(SubsampledSignal):
    """
    This is a Subsampled signal object, except it implements the unimplemented 'subsample' function.
    """
    def __init__(self, **kwargs):
        self.q = kwargs["q"]
        self.n = kwargs["n"]
        self.locq = kwargs["locq"]
        self.noise_sd = kwargs["noise_sd"]
        freq_normalized = 2j * np.pi * kwargs["locq"] / kwargs["q"]
        strengths = kwargs["strengths"]

        def sampling_function(query_batch):
            query_indices_qary_batch = np.array(dec_to_qary_vec(query_batch, self.q, self.n)).T
            return np.exp(query_indices_qary_batch @ freq_normalized) @ strengths

        self.sampling_function = sampling_function

        super().__init__(**kwargs)

    def subsample(self, query_indices):
        """
        Computes the signal/function values at the queried indicies on the fly
        """
        batch_size = 10000
        res = []
        query_indices_batches = np.array_split(query_indices, len(query_indices)//batch_size + 1)
        with Pool() as pool:
            for new_res in pool.imap(self.sampling_function, query_indices_batches):
                res = np.concatenate((res, new_res))
        return res

    def get_MDU(self, ret_num_subsample, ret_num_repeat, b, trans_times=False):
        """
        wraps get_MDU method from SubsampledSignal to add synthetic noise
        """
        mdu = super().get_MDU(ret_num_subsample, ret_num_repeat, b, trans_times)
        for i in range(len(mdu[2])):
            for j in range(len(mdu[2][i])):
                size = np.array(mdu[2][i][j]).shape
                nu = self.noise_sd / np.sqrt(2 * self.q ** b)
                mdu[2][i][j] += np.random.normal(0, nu, size=size + (2,)).view(np.complex).reshape(size)
        return mdu
