'''
Class for common interface to an input signal.
'''
from typing import Optional, Any

import numpy as np
import random
from src.qspright.utils import fwht, gwht_tensored, igwht_tensored, random_signal_strength_model, save_data, load_data
from pathlib import Path

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
        self._set_params(**kwargs)
        self._init_signal()

    def _set_params(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.noise_sd = kwargs.get("noise_sd", 0)
        self.N = self.q ** self.n
        self.signal_t = kwargs.get("signal_t")
        self.signal_w = kwargs.get("signal_w")
        self.calc_w = kwargs.get("calc_w", False)
        self.foldername = kwargs.get("folder")

    def _init_signal(self):

        if self.signal_t is None:
            signal_path = Path(f"{self.foldername}/signal_t.pickle")
            if signal_path.is_file():
                self.signal_t = load_data(Path(f"{self.foldername}/signal_t.pickle"))
            else:
                self.sample()
                Path(f"{self.foldername}").mkdir(exist_ok=True)
                save_data(self.signal_t, Path(f"{self.foldername}/signal_t.pickle"))

        if self.calc_w and self.signal_w is None:
            self.signal_w = gwht_tensored(self.signal_t, self.q, self.n)
            if np.linalg.norm(self.signal_t - igwht_tensored(self.signal_w, self.q, self.n))/self.N < 1e-5:
                print("verified transform")

    def sample(self):
        raise NotImplementedError

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
    def get_time_domain(self, base_inds):
        base_inds = np.array(base_inds)
        if len(base_inds.shape) == 3:
            return [self.signal_t[tuple(inds)] for inds in base_inds]
        elif len(base_inds.shape) == 2:
            return self.signal_t[tuple(base_inds)]
