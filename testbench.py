import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import galois
import sys
import copy
sys.path.append("src")
from src.utils import fwht, gwht, igwht, qary_ints, dec_to_bin, bin_to_dec, binary_ints, qary_vec_to_dec, \
    nth_roots_unity, dec_to_qary_vec, near_nth_roots
from src.query import compute_delayed_wht, compute_delayed_gwht, get_Ms, get_b, get_D
from src.reconstruct import singleton_detection
from src.inputsignal import Signal

debug = True
verbose = True
Us = []
Ss = []
np.random.seed(10)

n = 3
q = 4
b = 2
eps = 1e-7
GF = galois.GF(q)
omega = np.exp(2j * np.pi / q)
non_zeros = [4, 6, 10, 15, 24, 37, 48, 54]

print("The Non-Zero Indicies are:")
print(non_zeros)
signal = Signal(n, non_zeros, q=q, strengths=[2, 4, 1, 1, 1, 3, 8, 1], noise_sd=0)

num_delays = signal.n + 1
Ms = get_Ms(n, b, q, 4, "complex")
Ms = [np.array(M) for M in Ms]
K = np.array(qary_ints(signal.n, signal.q))
select_froms = [(M.T @ K) % q for M in Ms]

for M in Ms:
    D = get_D(signal.n, method="complex", num_delays=num_delays, q=q)
    D = np.array(D)
    print(D)
    U, used_inds = compute_delayed_gwht(signal, M, D, q)
    Us.append(U)
    Ss.append(omega ** np.array(D @ K))

Up = copy.deepcopy(Us)
num_peeling = 0
peeled = set([])
result = []
there_were_multitons = True
peeling_max = 8
max_iter = 3
iter_step = 0
while there_were_multitons and num_peeling < peeling_max and iter_step < max_iter:
    iter_step = iter_step + 1
    multitons = []
    singletons = {}
    for i, (U, S, select_from) in enumerate(zip(Us, Ss, select_froms)):
        for j, col in enumerate(U.T):
            j_qary = np.array(dec_to_qary_vec(np.array([j]), q, b))[:, 0]
            if debug:
                active_k_idx = []
                for idx in range(select_from.shape[1]):
                    if np.all(j_qary == select_from[:, idx]):
                        active_k_idx.append(idx)
                k_active = K[:, active_k_idx]
            print("For M(" + str(i) + ") entry U(" + str(j) + ") the active indicies are:")
            print(active_k_idx)
            print("The active and non-zero (unpeeled) indicies are:")
            active_non_zero = list(set(active_k_idx).intersection(non_zeros).difference(peeled))
            print(active_non_zero)
            if np.vdot(col, col) > eps:
                ratios = col[0] / col
                plt.title('Active and non-zero indices: ' + str(active_non_zero) + "  ({0},{1}), iter={2}".format(i,j,iter_step))
                plt.plot(np.real(ratios), np.imag(ratios), '*')
                plt.show()
                is_singleton = near_nth_roots(ratios, q, eps)
                if is_singleton:
                    singleton_ind = ((np.arange(q) @ (np.abs(ratios - np.outer(nth_roots_unity(q), np.ones(n+1))) < eps))[1:])
                    dec_singleton_ind = qary_vec_to_dec(singleton_ind, q)
                    rho = np.vdot(S[:, dec_singleton_ind], col) / len(col)
                    singletons[(i, j)] = (singleton_ind, rho)
                    print("We predict that the singleton index is " + str(dec_singleton_ind))
                if is_singleton:
                    print("We have a singleton!")
                else:
                    print("We have a multiton!")
                    multitons.append((i, j))
            else:
                print("We have a zeroton!")

    # Peeling Decoder
    if len(multitons) == 0:  # no more multitons, and can construct final WHT
        there_were_multitons = False

    # balls to peel
    balls_to_peel = set()
    ball_values = {}
    ball_sgn = {}
    for (i, j) in singletons:
        (k, rho) = singletons[(i, j)]
        ball = qary_vec_to_dec(k, q)
        balls_to_peel.add(ball)
        ball_values[ball] = rho
        result.append((k, ball_values[ball]))
    if verbose:
        print('These balls will be peeled')
        print(balls_to_peel)
    # peel
    for ball in balls_to_peel:
        num_peeling += 1
        peeled.add(ball)
        k = np.array(dec_to_qary_vec(np.array([ball]), q, n))
        print("Processing Singleton {0}".format(ball))
        print(k)
        potential_peels = [(l, qary_vec_to_dec(M.T.dot(k) % q, q)) for l, M in enumerate(Ms)]
        for (l, j) in potential_peels:
            print("The singleton appears in M({0}), U({1})".format(l, j))

        for peel in potential_peels:
            signature_in_stage = Ss[peel[0]][:, ball]
            to_subtract = ball_values[ball] * signature_in_stage.reshape(-1, 1)
            if verbose:
                print('this is subtracted:')
                print(to_subtract)
                print('from')
                print(Us[peel[0]][:, peel[1]])
                print("Peeled ball {0} off bin {1}".format(qary_vec_to_dec(k, q), peel))
            Us[peel[0]][:, peel[1]] -= to_subtract
        print("Iteration Complete: The peeled indicies are:")
        print(peeled)

gwht = np.zeros_like(signal.signal_t)
for k, value in result:
    idx = qary_vec_to_dec(k, q)
    if gwht[idx] == 0:
        gwht[idx] = value
gwht *= np.sqrt(b/n)
