'''
Methods for the query generator: specifically, to

1. generate sparsity coefficients b and subsampling matrices M
2. get the indices of a signal subsample
3. compute a subsampled and delayed Walsh-Hadamard transform.
'''

import numpy as np

from qspright.utils import fwht, gwht, bin_to_dec, qary_vec_to_dec, binary_ints, qary_ints


def get_Ms_simple(n, b, q, num_to_get=None):
    '''
    A semi-arbitrary fixed choice of the subsampling matrices. See get_Ms for full signature.
    '''
    Ms = []
    for i in range(num_to_get - 1, -1, -1):
        M = np.zeros((n, b), dtype=np.int32)
        M[(b * i) : (b * (i + 1)), :] = np.eye(b)
        Ms.append(M)

    return Ms


def get_Ms_complex(n, b, q, num_to_get=None):

    Ms = []
    # TODO Prevent duplicate M
    for i in range(num_to_get):
        M = np.random.randint(q, size=(n, b))
        Ms.append(M)
    return Ms

def get_Ms(n, b, q, num_to_get=None, method="simple"):
    '''
    Gets subsampling matrices for different sparsity levels.

    Arguments
    ---------
    n : int
    log2 of the signal length.

    b : int
    Sparsity.

    num_to_get : int
    The number of M matrices to return.

    method : str
    The method to use. All methods referenced must use this signature (minus "method".)

    Returns
    -------
    Ms : list of numpy.ndarrays, shape (n, b)
    The list of subsampling matrices.
    '''
    if num_to_get is None:
        num_to_get = max(n // b, 3)

    if method == "simple" and num_to_get > n // b:
        raise ValueError("When query_method is 'simple', the number of M matrices to return cannot be larger than n // b")

    return {
        "simple": get_Ms_simple,
        "complex": get_Ms_complex
    }.get(method)(n, b, q, num_to_get)

def get_D_identity(n, **kwargs):
    q=kwargs.get("q")
    int_delays = np.zeros(n, )
    for i in range(1): # Previously, q-1, but should just be 1
        int_delays = np.vstack((int_delays, (i+1)*np.eye(n)))
    return int_delays.astype(int)

def get_D_random(n, **kwargs):
    '''
    Gets a random delays matrix of dimension (num_delays, n). See get_D for full signature.
    '''
    q=kwargs.get("q")
    num_delays = kwargs.get("num_delays")
    return np.random.choice(q, (num_delays, n))

def get_D_nso(n, **kwargs):
    '''
    Get a repetition code based (NSO-SPRIGHT) delays matrix. See get_D for full signature.
    '''
    num_delays = kwargs.get("num_delays")
    q=kwargs.get("q")
    p1 = num_delays // (n + 1) # is this what we want?
    random_offsets = get_D_random(n, q=q, num_delays=p1)
    D = np.empty((0, n), dtype=int)
    identity_like = get_D_identity(n)
    for row in random_offsets:
        modulated_offsets = (row - identity_like) % q
        D = np.vstack((D, modulated_offsets))
    return D
    
def get_D(n, method="random", **kwargs):
    '''
    Delay generator: gets a delays matrix.

    Arguments
    ---------
    n : int
    number of bits: log2 of the signal length.

    Returns
    -------
    D : numpy.ndarray of binary ints, dimension (num_delays, n).
    The delays matrix; if num_delays is not specified in kwargs, see the relevant sub-function for a default.
    '''
    return {
        "random" : get_D_random,
        "identity" : get_D_identity,
        "nso" : get_D_nso
    }.get(method)(n, **kwargs)

def subsample_indices(M, d):
    '''
    Query generator: creates indices for signal subsamples.
    
    Arguments
    ---------    
    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.
    
    d : numpy.ndarray, shape (n,)
    The subsampling offset; takes on binary values.
    
    Returns
    -------
    indices : numpy.ndarray, shape (B,)
    The (decimal) subsample indices. Mostly for debugging purposes.
    '''
    L = binary_ints(M.shape[1])
    inds_binary = np.mod(np.dot(M, L).T + d, 2).T 
    return bin_to_dec(inds_binary)

def compute_delayed_gwht(signal, M, D, q):
    b = M.shape[1]
    L = np.array(qary_ints(b, q))  # List of all length b qary vectors
    base_inds = [(M @ L + np.outer(d, np.ones(q ** b, dtype=int))) % q for d in D]
    base_inds_dec = [qary_vec_to_dec(A, q) for A in base_inds]
    used_inds = np.swapaxes(np.array(base_inds), 0, 1)
    used_inds = np.reshape(used_inds, (used_inds.shape[0], -1))
    samples_to_transform = [signal.get_time_domain(tuple(inds)) for inds in base_inds]
    return np.array([gwht(row, q, b) for row in samples_to_transform]), used_inds

def compute_delayed_wht(signal, M, D):
    '''
    Creates random delays, subsamples according to M and the random delays,
    and returns the subsample WHT along with the delays.

    Arguments
    ---------
    signal : Signal object
    The signal to subsample, delay, and compute the WHT of.

    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.

    num_delays : int
    The number of delays to apply; or, the number of rows in the delays matrix.

    force_identity_like : boolean
    Whether to make D = [0; I] like in the noiseless case; for debugging.
    '''
    inds = np.array([subsample_indices(M, d) for d in D])
    used_inds = set(np.unique(inds))
    samples_to_transform = signal.signal_t[np.array([subsample_indices(M, d) for d in D])] # subsample to allow small WHTs
    return np.array([fwht(row) for row in samples_to_transform]), used_inds # compute the small WHTs
    