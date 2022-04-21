'''
Methods for the reconstruction engine; specifically, to

1. carry out singleton detection
2. get the cardinalities of all bins in a subsampling group (debugging only).
'''

import numpy as np
from utils import qary_vec_to_dec, dec_to_qary_vec

def singleton_detection_noiseless(U_slice, **kwargs):
    '''
    Finds the true index of a singleton, or the best-approximation singleton of a multiton.
    Assumes P = n + 1 and D = [0; I].
    
    Arguments
    ---------
    U_slice : numpy.ndarray, (P,).
    The WHT component of a subsampled bin, with element i corresponding to delay i.
    
    Returns
    -------
    k : numpy.ndarray
    Index of the corresponding right node, in binary form.
    '''
    return (-np.sign(U_slice * U_slice[0])[1:] == 1).astype(np.int), 1

def singleton_detection_mle(U_slice, **kwargs):
    '''
    Finds the true index of a singleton, or the best-approximation singleton of a multiton, in the presence of noise.
    Uses MLE: looks at the residuals created by peeling off each possible singleton.
    
    Arguments
    ---------
    U_slice : numpy.ndarray, (P,).
    The WHT component of a subsampled bin, with element i corresponding to delay i.

    selection : numpy.ndarray.
    The decimal preimage of the bin index, i.e. the list of potential singletons whose signature under M could be the j of the bin.

    S_slice : numpy.ndarray
    The set of signatures under the delays matrix D associated with each of the elements of 'selection'.

    n : int
    The signal's number of bits.

    Returns
    -------
    k : numpy.ndarray, (n,)
    The index of the singleton.

    '''
    selection, S_slice, q, n = kwargs.get("selection"), kwargs.get("S_slice"), kwargs.get("q"), kwargs.get("n")
    P = S_slice.shape[0]
    alphas = 1/P * np.dot(np.conjugate(S_slice).T, U_slice)
    residuals = np.linalg.norm(U_slice - (alphas * S_slice).T, ord=2, axis=1)
    k_sel = np.argmin(residuals)
    return selection[k_sel]


def singleton_detection(U_slice, method="mle", **kwargs):
    return {
        "mle" : singleton_detection_mle,
        "noiseless" : singleton_detection_noiseless,
    }.get(method)(U_slice, **kwargs)