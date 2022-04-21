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
    def __init__(self, n, loc, q=2, strengths=None, noise_sd=0):
        self.n = n
        self.q = q
        self.loc = loc # only for debug
        self.strengths = strengths
        self.noise_sd = noise_sd
        N = q ** n
        if strengths is None:
            strengths = np.ones_like(loc)
        wht = np.zeros((N,))
        for l, s in zip(loc, strengths):
            wht[l] = s
        if q == 2:
            wht = wht + np.random.normal(0, noise_sd, (N,))
            self.signal_t = fwht(wht)
            self.signal_w = fwht(self.signal_t) / N
        else:
            wht = wht + np.random.normal(0, noise_sd, (N,))
            self.signal_t = igwht(wht, q, n)
            self.signal_w = gwht(self.signal_t, q, n)
