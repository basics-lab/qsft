'''
Class for common interface to an input signal.
'''
from typing import Optional, Any

import numpy as np
import random
from src.qspright.utils import fwht, gwht_tensored, igwht_tensored


def random_signal_strength_model(sparsity, a, b):
    magnitude = np.random.uniform(a, b, sparsity)
    phase = np.random.uniform(0, 2*np.pi, sparsity)
    return magnitude * np.exp(1j*phase)


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
    '''

    def __init__(self, **kwargs):
        self._init_standard_params(**kwargs)
        if kwargs.get("signal") is None:
            self._init_random(**kwargs)
        else:
            self._init_given(**kwargs)

    def _init_standard_params(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.noise_sd = kwargs.get("noise_sd", 0)
        self.N = self.q ** self.n

    def _init_given(self, **kwargs):
        self.sparsity = kwargs.get("sparsity", 100)
        self._signal_t = np.reshape(kwargs.get("signal"), [self.q] * self.n)
        if kwargs.get("calc_w", False):
            self.signal_w = gwht_tensored(self._signal_t, self.q, self.n)
            if np.linalg.norm(self._signal_t - igwht_tensored(self.signal_w, self.q, self.n))/self.N < 1e-5:
                print("verified transform")


    def _init_random(self, **kwargs):
        self.sparsity = kwargs.get("sparsity")
        self.loc = random.sample(range(self.N), self.sparsity)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.strengths = random_signal_strength_model(self.sparsity, self.a, self.b)
        wht = np.zeros((self.N,), dtype=complex)
        for l, s in zip(self.loc, self.strengths):
            wht[l] = s
        self._signal_w = wht + np.random.normal(0, self.noise_sd, size=(self.N, 2)).view(np.complex).reshape(self.N)
        self._signal_w = np.reshape(self._signal_w, [self.q] * self.n)
        if self.q == 2:
            self._signal_t = fwht(self._signal_w)
            self._signal_t = np.reshape(self._signal_t, [self.q] * self.n)
        else:
            self._signal_t = igwht_tensored(self._signal_w, self.q, self.n)
            #if np.linalg.norm(self._signal_w - gwht_tensored(self._signal_t, self.q, self.n)) < 1e-3:
            #    print("verified transform")

    '''
    shape: returns the shape of the time domain signal.
    
    Returns
    -------
    shape of time domain signal
    '''
    def shape(self):
        return tuple([self.q for i in range(self.n)])

    '''
    Arguments
    ---------    
    inds: tuple of 1d n-element arrays that represent the indicies to be queried
    
    Returns
    -------
    indices : linear output of the queried indicies
    '''
    def get_time_domain(self, inds):
        return self._signal_t[inds]
