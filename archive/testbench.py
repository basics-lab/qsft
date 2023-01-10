import numpy as np
import sys
import time
from synt_exp.synt_src.synthetic_signal import get_random_signal
from qsft.qsft import QSFT

np.random.seed(10)
q = 4
n = 10
N = q ** n
sparsity = 200
a_min = 1
a_max = 10
noise_sd = 0

test_signal = get_random_signal(n=n, q=q, sparsity=sparsity, a_min=a_min, a_max=a_max, noise_sd=noise_sd)
print("test signal generated")

transformer = QSFT(
    query_method="complex",
    delays_method="identity",
    reconstruct_method="noiseless",
    num_subsample=3,
    b=4
)
start_time = time.time()
gwht = transformer.transform(test_signal, verbose=False, report=False)
#print(f"SPRIGHT Runtime:{time.time() - start_time}")
#gwht_lasso, non_zero = lasso_decode(test_signal, 0.3, refine=True, verbose=True)

#print("found non-zero indices SPRIGHT: ")
#print(np.sort(peeled))
#print("found non-zero indices LASSO: ")
#print(np.sort(non_zero))
#print("true non-zero indices: ")
#print(np.sort(test_signal.locq.T))

#print("total sample ratio = ", n_used / q ** n)
#print("unique sample ratio = ", n_used_unique / q ** n)

print("NMSE SPRIGHT= ", np.sum(np.abs(test_signal.signal_w - gwht)**2) / np.sum(np.abs(test_signal.signal_w)**2))
#print("NMSE LASSO= ", np.sum(np.abs(test_signal._signal_w - gwht_lasso)**2) / np.sum(np.abs(test_signal._signal_w)**2))