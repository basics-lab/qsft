import numpy as np


import sys
sys.path.append("./src/qspright/")

from src.qspright.inputsignal import Signal
from src.qspright.qspright_nso import QSPRIGHT

np.random.seed(10)

q = 4
n = 10
N = q ** n
num_nonzero_indices = 50
nonzero_indices = np.random.choice(N, num_nonzero_indices, replace=False)
nonzero_values = 2 + 3 * np.random.rand(num_nonzero_indices)
nonzero_values = nonzero_values * np.exp(1j * 2 * np.pi * np.random.rand(num_nonzero_indices))
noise_sd = 0.03

test_signal = Signal(n=n, q=q, loc=nonzero_indices, strengths=nonzero_values, noise_sd=noise_sd)
print("test signal generated")

spright = QSPRIGHT(
    query_method="complex",
    delays_method="nso",
    reconstruct_method="nso",
    num_subsample=5,
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

print("NMSE = ", np.sum(np.abs(test_signal._signal_w - gwht)**2) / np.sum(np.abs(test_signal._signal_w)**2))