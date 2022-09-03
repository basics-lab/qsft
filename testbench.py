import numpy as np
import sys
sys.path.append("./src/qspright/")
from src.qspright.inputsignal import Signal
from src.qspright.qspright_nso import QSPRIGHT
from src.qspright.utils import lasso_decode

np.random.seed(10)
q = 3
n = 6
N = q ** n
sparsity = 20
a = 1
b = 10
noise_sd = 0.01

test_signal = Signal(n=n, q=q, sparsity=sparsity, a=a, b=b, noise_sd=noise_sd)
print("test signal generated")

spright = QSPRIGHT(
    query_method="simple",
    delays_method="identity",
    reconstruct_method="noiseless",
    num_subsample=2,
    b=3
)

gwht, (n_used, n_used_unique, _), peeled = spright.transform(test_signal, verbose=False, report=True)
gwht_lasso, non_zero = lasso_decode(test_signal, 0.30)

print("found non-zero indices: ")
print(np.sort(peeled))
print("found non-zero indices LASSO: ")
print(np.sort(non_zero))
print("true non-zero indices: ")
print(np.sort(test_signal.loc))

print("total sample ratio = ", n_used / q ** n)
print("unique sample ratio = ", n_used_unique / q ** n)
gwht_reshaped = np.reshape(gwht, -1)
print("NMSE = ", np.sum(np.abs(test_signal._signal_w - gwht)**2) / np.sum(np.abs(test_signal._signal_w)**2))