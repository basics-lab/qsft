import numpy as np


import sys
sys.path.append("./src/qspright/")

from src.qspright.inputsignal import Signal
from src.qspright.qspright_nso import QSPRIGHT

np.random.seed(10)

q = 4
n = 10
N = q ** n
num_nonzero_indices = 150
nonzero_indices = np.random.choice(N, num_nonzero_indices, replace=False)
nonzero_values = 2 + 3 * np.random.rand(num_nonzero_indices)
nonzero_values = nonzero_values * (2 * np.random.binomial(1, 0.5, size=num_nonzero_indices) - 1)
noise_sd = 0.01

test_signal = Signal(n=n, q=q, loc=nonzero_indices, strengths=nonzero_values, noise_sd=noise_sd)
print("test signal generated")

spright = QSPRIGHT(
    query_method="simple",
    delays_method="nso",
    reconstruct_method="nso",
    num_subsample=2,
    num_random_delays=5,
    b=5
)

gwht, (n_used, n_used_unique), peeled = spright.transform(test_signal, verbose=False, report=True)

print("found non-zero indices: ")
print(np.sort(peeled))

print("true non-zero indices: ")
print(np.sort(nonzero_indices))

print("total sample ratio = ", n_used / q ** n)
print("unique sample ratio = ", n_used_unique / q ** n)


gwht_reshaped = np.reshape(gwht, -1)

print("NMSE = ", np.sum(np.abs(test_signal.signal_w - gwht_reshaped)**2) / np.sum(np.abs(test_signal.signal_w)**2))