import numpy as np
import sys

sys.path.append("./src/qspright/")
from src.qspright.qspright_precomputed import QSPRIGHT
from src.qspright.input_signal_precomputed import PrecomputedSignal
from src.rna_transform.input_rna_signal_precomputed import PrecomputedSignalRNA

if __name__ == '__main__':
    np.random.seed(20)
    q = 4
    n = 8
    N = q ** n
    sparsity = 20
    a_min = 1
    a_max = 10
    b = 3
    noise_sd = 1e-5

    query_args = {
        "query_method": "complex",
        "delays_method": "nso",
        "num_subsample": 2,
        "num_random_delays": 5,
        "b": b
    }
    """
    Use the constructor like this to save the computation you've done
    If you do not pass a `signal` argument, we assume you want to randomly generate the signal
    """
    test_signal = PrecomputedSignal(n=n,
                                    q=q,
                                    sparsity=sparsity,
                                    a_min=a_min,
                                    a_max=a_max,
                                    noise_sd=noise_sd,
                                    query_args=query_args)

    """
    The PrecomputeSignalRNA Class is only slightly different from the standard PrecomputedSignal Class, If constructed without 
    the signal keyword argument, the PrecomputedSignalRNA object is intended to be used to save samples. When the 
    .sample() function is called, it will be use the ViennaRNA package to generate the data, with subsampling patten 
    chosen based on the query args. Note that you must include the positions argument, which should be a list of
    integers < q ** n of length n
    """
    test_signal_RNA = PrecomputedSignalRNA(n=n,
                                           q=q,
                                           noise_sd=noise_sd,
                                           positions = [5, 10, 15, 20, 25, 30, 35, 40],
                                           query_args=query_args)
    """
    subsample() computes the subsamples and saves the output into a folder "foldername". By default, for each M matrix, a 
    .pickle file is saved, representing all the samples corresponding to that M
    all_b - If you want to save sampling patterns for b >=2 up to the value of b passed in the construction
    save_locally - If you actually want to use this signal object to run qspright, set this to True, otherwise this 
                  should typically be be False.
    Parallelization should be implemented, but has not been yet.
    subsample_nosave() should work as well, and should be faster? (not tested)
    """
    test_signal.subsample(foldername="test1", save_all_b=False, keep_samples=True)
    test_signal_RNA.subsample(foldername="test_RNA", save_all_b=False, keep_samples=True)
    """
    If you have set save_locally = True, or you ran subsample_nosave(), you can still save all the subsampled entries in
    a single file
    """
    test_signal.save_full_signal("full_signal.pickle")
    """
    This function can be called if the signal has a transform variable _signal_w that you want to save
    """
    test_signal.save_transform("saved_tf.pickle")
    print("test signal generated")
    """
    We can load a signal from a folder. The "M_select" list specifies which Ms are to be used
    If the b value is provided, it is assumed that the data was generated with all_b=True, if this is not
    the case, the b value must not be provided (and b is inferred from the saved Ms).
    """
    # test_signal = PrecomputedSignal(signal="test1",
    #                                 M_select=[True, True, True, True, True],
    #                                 noise_sd=noise_sd,
    #                                 transform="saved_tf.pickle",
    #                                 b=4)
    """
    We can also load the signal from a file, by providing the transform field, we can also load the transform in file mode,
    just as we did above, but we do not do that here.
    """
    # test_signal_from_file = PrecomputedSignal(signal="full_signal.pickle",
    #                                           noise=noise_sd)
    qspright_args = {
        "num_subsample": 2,
        "num_random_delays": 5,
        "b": b,
        "noise_sd": noise_sd
    }

    spright = QSPRIGHT("nso", **qspright_args)

    result = spright.transform(test_signal, verbose=False, timing_verbose=True,
                                                                 report=True)

    gwht = result.get("gwht")
    n_used = result.get("n_samples")
    n_used_unique = result.get("n_unique_samples")
    peeled = result.get("locations")
    avg_hamming_weight = result.get("avg_hamming_weight")

    # gwht_lasso, non_zero = lasso_decode(test_signal, 0.30)

    print("found non-zero indices SPRIGHT: ")
    print(peeled)
    # print("found non-zero indices LASSO: ")
    # print(np.sort(non_zero))
    print("true non-zero indices: ")
    print(test_signal.get_nonzero_locations())

    print("total sample ratio = ", n_used / q ** n)
    print("unique sample ratio = ", n_used_unique / q ** n)

    signal_w_diff = test_signal._signal_w.copy()

    for key in gwht.keys():
        signal_w_diff[key] = signal_w_diff.get(key, 0) - gwht[key]
    print("NMSE SPRIGHT= ",
          np.sum(np.abs(list(signal_w_diff.values())) ** 2) / np.sum(np.abs(list(test_signal._signal_w.values())) ** 2))

    print("AVG Hamming Weight of Nonzero Locations = ", avg_hamming_weight)
    #print("NMSE LASSO= ", np.sum(np.abs(test_signal._signal_w - gwht_lasso)**2) / np.sum(np.abs(test_signal._signal_w)**2))