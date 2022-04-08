import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import galois
import sys
sys.path.append("src")
from src.utils import fwht, gwht, igwht, qary_ints, dec_to_bin, bin_to_dec, binary_ints, qary_vec_to_dec,nth_roots_unity, dec_to_qary_vec
from src.query import compute_delayed_wht, compute_delayed_gwht, get_Ms, get_b, get_D
from src.reconstruct import singleton_detection
from src.inputsignal import Signal
debug = True
Us=[]
n = 4
q = 3
b = 2
eps = 0.1
GF = galois.GF(q)
non_zeros = [4, 6, 10, 15, 24, 37, 48, 54]
print("The Non-Zero Indicies are:")
print(non_zeros)
signal = Signal(n, non_zeros, q=q, strengths=[2, 4, 1, 1, 1, 3, 8, 1], noise_sd=0)
num_delays = signal.n + 1
Ms = get_Ms(n, b, q, 1, "complex")
K = GF(qary_ints(signal.n, signal.q))
select_froms = [-M.T @ K for M in Ms]
for M in Ms:
    D = get_D(signal.n, method="complex", num_delays=num_delays, q=q)
    L = GF(qary_ints(b, q))
    base_inds = [M @ L + np.outer(d, GF.Ones(q ** b)) for d in D]
    base_inds_dec = [qary_vec_to_dec(A, q) for A in base_inds]
    used_inds = set(np.unique(base_inds))
    samples_to_transform = signal.signal_t[np.array(base_inds_dec)]
    U = np.array([gwht(row, q, b) for row in samples_to_transform])
    Us.append(U)
print("Searching for Singletons")
for i, (U, select_from) in enumerate(zip(Us, select_froms)):
    for j, col in enumerate(U.T):
        j_qary = GF(dec_to_qary_vec(np.array([j]), q, b))[:,0]
        print(j_qary)
        if debug:
            active_k_idx = []
            for idx in range(select_from.shape[1]):
                if (j_qary == select_from[:, idx]).all():
                    active_k_idx.append(idx)
            k_active = K[:, active_k_idx]
        print("For M(" + str(i) +") entry " + str(j)+ " the active indicies are:")
        print(active_k_idx)
        print("The active and non-zero indicies are:")
        active_non_zero = list(set(active_k_idx).intersection(non_zeros))
        print(active_non_zero)
        ratios = col[0] / col
        plt.plot(np.real(ratios), np.imag(ratios), '*')
        plt.show()
        in_set = np.zeros(ratios.shape, dtype=bool)
        omega = nth_roots_unity(q)
        for l in range(q):
            in_set = in_set | (np.square(np.abs(ratios - omega[l])) < eps)
        is_singleton = in_set.all()
        if is_singleton:
            print("We have a singleton!")
