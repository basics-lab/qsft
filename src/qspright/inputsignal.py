'''
Class for common interface to an input signal.
'''

import numpy as np
from utils import fwht, gwht, igwht

class Signal:
    '''
    Class to encapsulate a time signal and its Walsh-Hadamard (W-H) transform.

    Attributes
    ---------
    n : int
    number of bits: log2 of the signal length.
    
    loc : iterable
    Locations of true peaks in the W-H spectrum. Elements must be integers in [0, q ** n - 1].
    
    strengths : iterable
    The strength of each peak in the W-H spectrum. Defaults to all 1s. Length has to match that of loc.
    
    noise_sd : scalar
    The standard deviation of the added noise.
    
    signal_t : numpy.ndarray
    The time signal.
    
    signal_w : numpy.ndarray
    The WHT of input_signal.
    '''
    def __init__(self, **kwargs):
        if kwargs.get("signal") is None:
            self._init_random(**kwargs)
        else:
            self._init_given(**kwargs)

    def _init_given(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.signal_t = kwargs.get("signal")
        self.signal_t_qidx = np.reshape(self.signal_t, [self.q] * self.n)
        self.noise_sd = kwargs.get("noise_sd")
        self.N = self.q ** self.n
        self.sparsity = kwargs.get("sparsity", 100)
        if kwargs.get("calc_w", False):
            self.signal_w = gwht(self.signal_t, self.q, self.n)
            if np.linalg.norm(self.signal_t - igwht(self.signal_w, self.q, self.n))/self.N < 1e-5:
                print("verified transform")


    def _init_random(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.noise_sd = kwargs.get("noise_sd", 0)
        self.N = self.q ** self.n
        self.loc = kwargs.get("loc")
        self.sparsity = len(self.loc)
        self.strengths = kwargs.get("strengths", np.ones_like(self.loc))
        wht = np.zeros((self.N,))
        for l, s in zip(self.loc, self.strengths):
            wht[l] = s
        self.signal_w = wht + np.random.normal(0, self.noise_sd, (self.N,))
        if self.q == 2:
            self.signal_t = fwht(self.signal_w)
            self.signal_t_qidx = np.reshape(self.signal_t, [self.q] * self.n)
        else:
            self.signal_t = igwht(self.signal_w, self.q, self.n)
            self.signal_t_qidx = np.reshape(self.signal_t, [self.q] * self.n)
            if np.linalg.norm(self.signal_w - gwht(self.signal_t, self.q, self.n))/self.N < 1e-5:
                print("verified transform")

