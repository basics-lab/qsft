import numpy as np
from group_lasso import GroupLasso
from sklearn.linear_model import Ridge
import time
from qspright.utils import dec_to_qary_vec, qary_ints, calc_hamming_weight


def lasso_decode(signal, n_samples, refine=False, verbose=False, report=True):
    q = signal.q
    n = signal.n
    N = q ** n
    dtype = int if (q ** 2)*n > 255 else np.uint8

    start_time = time.time()
    if verbose:
        print("Setting up LASSO problem")

    (sample_idx_dec, y) = list(signal.signal_t.keys()), list(signal.signal_t.values())
    sample_idx_dec = sample_idx_dec[:n_samples]
    y = y[:n_samples]
    sample_idx = dec_to_qary_vec(sample_idx_dec, q, n, dtype=dtype)
    y = np.concatenate((np.real(y), np.imag(y)))
    freqs = np.array(sample_idx).T @ qary_ints(n, q, dtype=dtype)
    X = np.exp(2j*np.pi*freqs/q).astype(np.csingle)
    X = np.concatenate((np.concatenate((np.real(X), -np.imag(X)), axis=1), np.concatenate((np.imag(X), np.real(X)), axis=1)))
    groups = [i % N for i in range(2*N)]

    if verbose:
        print(f"Setup Time:{time.time() - start_time}sec")
        print("Running Iterations...")
        start_time = time.time()

    lasso = GroupLasso(groups=groups,
                       group_reg=0.1,
                       l1_reg=0,
                       tol=1e-5,
                       n_iter=1000,
                       supress_warning=True,
                       fit_intercept=False)
    lasso.fit(X, y)

    if verbose:
        print(f"LASSO fit time:{time.time() - start_time}sec")

    w = lasso.coef_
    non_zero = np.nonzero(w[:N, 0])[0]
    if refine:
        ridge = Ridge(alpha=0.1, tol=1e-8)
        ridge.fit(X[:, non_zero], y)
        w[non_zero] = ridge.coef_[:, np.newaxis]
    gwht = w[0:N] + 1j*w[N:(2*N)]

    gwht = np.reshape(gwht, [q] * n)

    gwht_dict = {}

    non_zero_pos = np.array(np.nonzero(gwht)).T
    for p in non_zero_pos:
        gwht_dict[tuple(p)] = gwht[tuple(p)]

    if not report:
        return gwht_dict
    else:
        if len(non_zero_pos) > 0:
            loc = list(non_zero_pos)
            avg_hamming_weight = np.mean(calc_hamming_weight(loc))
            max_hamming_weight = np.max(calc_hamming_weight(loc))
        else:
            loc, avg_hamming_weight, max_hamming_weight = [], 0, 0
        result = {
            "gwht": gwht_dict,
            "n_samples": n_samples,
            "locations": loc,
            "avg_hamming_weight": avg_hamming_weight,
            "max_hamming_weight": max_hamming_weight
        }
        return result