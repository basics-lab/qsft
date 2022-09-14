import numpy as np
import sys

sys.path.append("./src/qspright/")
from src.qspright.qspright_precomputed import QSPRIGHT
from src.qspright.input_signal_precomputed import PrecomputedSignal

if __name__ == '__main__':
    np.random.seed(10)
    q = 4
    n = 20
    N = q ** n
    sparsity = 200
    a_min = 1
    a_max = 10
    b = 4
    noise_sd = 1e-5
    query_args = {
        "query_method": "complex",
        "delays_method": "nso",
        "num_subsample": 3,
        "num_random_delays": 3,
        "b": b
    }

    test_signal = PrecomputedSignal(n=n,
                                    q=q,
                                    sparsity=sparsity,
                                    a_min=a_min,
                                    a_max=a_max,
                                    noise_sd=noise_sd,
                                    query_args=query_args)

    test_signal.subsample()
    test_signal.save_signal("saved_signal.pickle")
    test_signal.save_transform("saved_tf.pickle")
    print("test signal generated")
    test_signal = 0
    test_signal = PrecomputedSignal(signal="saved_signal.pickle", noise_sd=noise_sd, transform="saved_tf.pickle")

    qspright_args = {
        "num_subsample": 3,
        "num_random_delays": 3,
        "b": b
    }

    spright = QSPRIGHT("nso", **qspright_args)

    gwht, (n_used, n_used_unique, _), peeled = spright.transform(test_signal, verbose=False, timing_verbose=True,
                                                                 report=True)

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
    print("NMSE SPRIGHT= ",
          np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(test_signal._signal_w.values())) ** 2))

    # print("NMSE LASSO= ", np.sum(np.abs(test_signal._signal_w - gwht_lasso)**2) / np.sum(np.abs(test_signal._signal_w)**2))
