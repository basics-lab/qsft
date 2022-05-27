import numpy as np

import sys
sys.path.append("src")

from qspright.inputsignal import Signal
from qspright.qspright_nso import QSPRIGHT

np.random.seed(10)

q = 4
n = 10
N = q ** n
num_nonzero_indices = 100
nonzero_indices = np.random.choice(N, num_nonzero_indices, replace=False)
nonzero_values = 2 + 3 * np.random.rand(num_nonzero_indices)
nonzero_values = nonzero_values * (2 * np.random.binomial(1, 0.5, size=num_nonzero_indices) - 1)
noise_sd = 1

test_signal = Signal(n=n, q=q, loc=nonzero_indices, strengths=nonzero_indices, noise_sd=noise_sd)
print("test signal generated")

spright = QSPRIGHT(
    query_method="complex",
    delays_method="nso",
    reconstruct_method="nso"
)

gwht, _, peeled = spright.transform(test_signal, verbose=False, report=True)

print("found non-zero indices: ")
print(np.sort(peeled))

print("true non-zero indices: ")
print(np.sort(nonzero_indices))
