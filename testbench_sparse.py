import numpy as np
import sys
sys.path.append("./src/qspright/")
from src.qspright.qspright_sparse import QSPRIGHT
from src.qspright.input_signal_long import LongSignal
from src.qspright.utils import lasso_decode, qary_vec_to_dec

if __name__ == '__main__':
    np.random.seed(10)
    q = 4
    n = 20
    N = q ** n
    sparsity = 200
    a = 1
    b = 10
    noise_sd = 1e-5

    test_signal = LongSignal(n=n, q=q, sparsity=sparsity, a=a, b=b, noise_sd=noise_sd)
    print("test signal generated")

    spright = QSPRIGHT(
        query_method="complex",
        delays_method="nso",
        reconstruct_method="noiseless",
        num_subsample=3,
        num_random_delays=3,
        b=4
    )

    gwht, (n_used, n_used_unique, _), peeled = spright.transform(test_signal, verbose=False, timing_verbose=True, report=True)
    # gwht_lasso, non_zero = lasso_decode(test_signal, 0.30)

    print("found non-zero indices SPRIGHT: ")
    print(np.sort(peeled))
    # print("found non-zero indices LASSO: ")
    # print(np.sort(non_zero))
    print("true non-zero indices: ")
    print(np.sort(test_signal.get_nonzero_locations()))

    print("total sample ratio = ", n_used / q ** n)
    print("unique sample ratio = ", n_used_unique / q ** n)

    signal_w_diff = test_signal._signal_w.copy()

    for key in gwht.keys():
        signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
    print("NMSE SPRIGHT= ", np.sum(np.abs(list(signal_w_diff.values()))**2) / np.sum(np.abs(list(test_signal._signal_w.values()))**2))

    # print("NMSE LASSO= ", np.sum(np.abs(test_signal._signal_w - gwht_lasso)**2) / np.sum(np.abs(test_signal._signal_w)**2))