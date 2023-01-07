import numpy as np
import tqdm
import sys
sys.path.append("../src")
from src.reconstruct import singleton_detection
from src.utils import qary_ints,  bin_to_dec, qary_vec_to_dec, dec_to_qary_vec
from src.query import compute_delayed_gwht, get_Ms, get_b, get_D
from src.inputsignal import get_random_signal

M_Golay = np.array(
    [[1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0],
     [1, 1, 2, 1, 0, 2, 0, 1, 0, 0, 0],
     [1, 2, 1, 0, 1, 2, 0, 0, 1, 0, 0],
     [1, 2, 0, 1, 2, 1, 0, 0, 0, 1, 0],
     [1, 0, 2, 2, 1, 1, 0, 0, 0, 0, 1]]).T
b = 5
n = 11
q = 3
omega = np.exp(2j * np.pi / q)
D = get_D(n, q=q, method="complex", num_delays=n+1)
K = np.array(qary_ints(n, q))
S = omega ** np.array(D @ K)

M = get_Ms(n, b, q, 1, "complex")[0]
select_from = qary_vec_to_dec(np.mod(M.T @ K, q), q)

signal, non_zeros = get_random_signal(n, q, q ** (b-1))
U, used_i = compute_delayed_gwht(signal, M, D, q)
for j, col in enumerate(U.T):
    if np.linalg.norm(col) ** 2 > cutoff:
        selection = np.where(select_from == j)[0]
        k_dec = singleton_detection(
            col,
            method="mle",
            selection=selection,
            S_slice=S[:, selection],
            q=signal.q,
            n=signal.n
        )  # find the best fit singleton

        k = np.array(dec_to_qary_vec([k_dec], signal.q, signal.n)).T[0]
        rho = np.dot(np.conjugate(S[:, k_dec]), col) / S.shape[0]
        residual = col - rho * S[:, k_dec]
if np.linalg.norm(residual) ** 2 > cutoff:
    multitons.append((i, j))
    if verbose:
        print("We have a Multiton")
else:  # declare as singleton
    singletons[(i, j)] = (k, rho)

