import time
import numpy as np
import tqdm
from src.qspright.reconstruct import singleton_detection
from src.qspright.utils import bin_to_dec, qary_vec_to_dec, sort_qary_vecs, calc_hamming_weight
from src.qspright.input_signal_long import LongSignal
from src.qspright.query import compute_delayed_gwht

class QSPRIGHT:
    """
    Class to store encoder/decoder configurations and carry out encoding/decoding.

    Attributes
    ----------
    query_method : str
    The method to generate the sparsity coefficient and the subsampling matrices.
    Currently implemented methods:
        "simple" : choose some predetermined matrices based on problem size.

    delays_method : str
    The method to generate the matrix of delays.
    Currently implemented methods:
        "identity_like" : return a zero row and the identity matrix vertically stacked together.
        "random" : make a random set of delays.

    reconstruct_method : str
    The method to detect singletons.
    Currently implemented methods:
        "noiseless" : decode according to [2], section 4.2, with the assumption the signal is noiseless.
        "mle" : naive noisy decoding; decode by taking the maximum-likelihood singleton that could be at that bin.
    """
    def __init__(self, reconstruct_method, **kwargs):
        self.reconstruct_method = reconstruct_method
        self.num_subsample = kwargs.get("num_subsample")
        self.num_random_delays = kwargs.get("num_random_delays")
        self.b = kwargs.get("b")
        self.noise_sd = kwargs.get("noise_sd")

    def transform(self, signal : LongSignal, verbosity=0, report=False, timing_verbose=False, **kwargs):
        '''
        Full SPRIGHT encoding and decoding. Implements Algorithms 1 and 2 from [2].
        (numbers) in the comments indicate equation numbers in [2].

        Arguments
        ---------
        signal : Signal object.
        The signal to be transformed / compared to.

        verbose : boolean
        Whether to print intermediate steps.

        Returns
        -------
        wht : ndarray
        The WHT constructed by subsampling and peeling.
        '''

        num_peeling = 0
        q = signal.q
        n = signal.n

        omega = np.exp(2j * np.pi / q)
        result = []

        gwht = {}
        gwht_counts = {}

        peeling_max = q ** n
        peeled = set([])

        b = self.b

        gamma = 1.5

        Ms, Ds = signal.get_MD(self.num_subsample, self.num_random_delays, self.b)
        Us = []
        used = []

        if verbosity >= 1:
            print("Calculating transforms...", flush=True)

        if timing_verbose:
            start_time = time.time()
        # subsample with shifts [D], make the observation [U]
        for (M, D) in zip(Ms, Ds):
            if verbosity >= 5:
                print("------")
                print("subsampling matrix")
                print(M)
                print("delay matrix")
                print(D)
            U = []
            used_sub = []
            for D_sub in D:
                U_sub, used_i = compute_delayed_gwht(signal, M, D_sub, q)
                U.append(U_sub)
                used_sub.append(used_i)
            Us.append(U)
            used.append(np.concatenate(used_sub, axis = 1))
        if timing_verbose:
            print(f"Transform Time:{time.time() - start_time}", flush=True)
        used = np.concatenate(used, axis=1)

        for i in range(len(Ds)):
            Us[i] = np.vstack(Us[i])
            Ds[i] = np.vstack(Ds[i])

        cutoff = 1e-9 + 2 * (1 + gamma) * (self.noise_sd ** 2) * (q ** (n - b))  # noise threshold
        cutoff = kwargs.get("cutoff", cutoff)

        if verbosity >= 2:
            print("cutoff = ", cutoff, flush=True)

        # begin peeling
        # index convention for peeling: 'i' goes over all M/U/S values
        # i.e. it refers to the index of the subsampling group (zero-indexed - off by one from the paper).
        # 'j' goes over all columns of the WHT subsample matrix, going from 0 to 2 ** b - 1.
        # e.g. (i, j) = (0, 2) refers to subsampling group 0, and aliased bin 2 (10 in binary)
        # which in the example of section 3.2 is the multiton X[0110] + X[1010] + W1[10]

        # a multiton will just store the (i, j)s in a list
        # a singleton will map from the (i, j)s to the true (binary) values k.
        # e.g. the singleton (0, 0), which in the example of section 3.2 is X[0100] + W1[00]
        # would be stored as the dictionary entry (0, 0): array([0, 1, 0, 0]).
        max_iter = 20
        iter_step = 0
        cont_peeling = True
        if timing_verbose:
            start_time = time.time()
        while cont_peeling and num_peeling < peeling_max and iter_step < max_iter:
            iter_step += 1
            if verbosity >= 2:
                print('-----')
                print("iter ", iter_step, flush=True)
                # print('the measurement matrix')
                # for U in Us:
                #     print(U)
            # first step: find all the singletons and multitons.
            singletons = {}  # dictionary from (i, j) values to the true index of the singleton, k.
            multitons = []  # list of (i, j) values indicating where multitons are.
            for i, (U, D) in enumerate(zip(Us, Ds)):
                for j, col in enumerate(U.T):
                    if np.linalg.norm(col) ** 2 > cutoff * len(col):

                        k = singleton_detection(
                            col,
                            method=self.reconstruct_method,
                            q=q,
                            n=n,
                            nso_subtype = "nso1"
                        )  # find the best fit singleton
                        #k = np.array(dec_to_qary_vec([k_dec], signal.q, signal.n)).T[0]
                        signature = omega ** (D @ k)
                        rho = np.dot(np.conjugate(signature), col) / D.shape[0]
                        residual = col - rho * signature

                        if verbosity >= 5:
                            print((i, j), np.linalg.norm(residual) ** 2, cutoff * len(residual))
                        if np.linalg.norm(residual) ** 2 > cutoff * len(residual):
                            multitons.append((i, j))
                            if verbosity >= 6:
                                print("We have a Multiton")
                        else:  # declare as singleton
                            singletons[(i, j)] = (k, rho)
                            if verbosity >= 3:
                                print("We have a Singleton at " + str(k))
                    else:
                        if verbosity >= 6:
                            print("We have a zeroton!")

            # all singletons and multitons are discovered
            if verbosity >= 5:
                print('singletons:')
                for ston in singletons.items():
                    print("\t{0} {1}\n".format(ston, bin_to_dec(ston[1][0])))

                print("Multitons : {0}\n".format(multitons))

            # raise RuntimeError("stop")
            # TODO WARNING: this is not a correct thing to do
            # in the last iteration of peeling, everything will be singletons and there
            # will be no multitons
            if len(multitons) == 0 or len(singletons) == 0: # no more multitons or no more singletons, and can construct final WHT
                cont_peeling = False

            # balls to peel
            balls_to_peel = set()
            ball_values = {}
            for (i, j) in singletons:
                k, rho = singletons[(i, j)]
                ball = tuple(k)  # Must be a hashable type
                #qary_vec_to_dec(k, q)
                balls_to_peel.add(ball)
                ball_values[ball] = rho
                result.append((k, ball_values[ball]))

            if verbosity >= 5:
                print('these balls will be peeled')
                print(balls_to_peel)
            # peel
            for ball in balls_to_peel:
                num_peeling += 1

                k = np.array(ball)[..., np.newaxis]
                potential_peels = [(l, qary_vec_to_dec(M.T.dot(k) % q, q)) for l, M in enumerate(Ms)]
                if verbosity >= 6:
                    k_dec = qary_vec_to_dec(k, q)
                    peeled.add(int(k_dec))
                    print("Processing Singleton {0}".format(k_dec))
                    print(k)
                    for (l, j) in potential_peels:
                        print("The singleton appears in M({0}), U({1})".format(l, j))
                for peel in potential_peels:
                    signature_in_stage = omega ** (Ds[peel[0]] @ k)
                    to_subtract = ball_values[ball] * signature_in_stage.reshape(-1, 1)
                    if verbosity >= 6:
                        # print('this is subtracted:')
                        # print(to_subtract)
                        # print('from')
                        # print(Us[peel[0]][:, peel[1]])
                        print("Peeled ball {0} off bin {1}".format(qary_vec_to_dec(k, q), peel))
                    Us[peel[0]][:, peel[1]] -= to_subtract

                if verbosity >= 5:
                    print("Iteration Complete: The peeled indicies are:")
                    print(np.sort(list(peeled)))

        loc = set()
        for k, value in result: # iterating over (i, j)s
            loc.add(tuple(k))
            if tuple(k) in gwht_counts:
                gwht[tuple(k)] = (gwht[tuple(k)] * gwht_counts[tuple(k)] + value) / (gwht_counts[tuple(k)] + 1)
                gwht_counts[tuple(k)] = gwht_counts[tuple(k)] + 1
            else:
                gwht[tuple(k)] = value
                gwht_counts[tuple(k)] = 1
        if timing_verbose:
            print(f"Peeling Time:{time.time() - start_time}", flush=True)

        if not report:
            return gwht
        else:
            used_unique = np.unique(used, axis=1)
            if len(loc) > 0:
                loc = sort_qary_vecs(list(loc))
                avg_hamming_weight = np.mean(calc_hamming_weight(loc))
                max_hamming_weight = np.max(calc_hamming_weight(loc))
            else:
                loc, avg_hamming_weight, max_hamming_weight = [], 0, 0
            result = {
                "gwht": gwht,
                "n_samples": np.shape(used)[-1],
                "n_unique_samples": np.shape(used_unique)[-1],
                "locations": loc,
                "avg_hamming_weight": avg_hamming_weight,
                "max_hamming_weight": max_hamming_weight
            }
            return result

    def method_test(self, signal, num_runs=10):
        '''
        Tests a method on a signal and reports its average execution time and sample efficiency.
        '''
        time_start = time.time()
        samples = 0
        successes = 0
        for i in tqdm.trange(num_runs):
            wht, num_samples, loc = self.transform(signal, report=True)
            if loc == set(signal.loc):
                successes += 1
            samples += num_samples
        return (time.time() - time_start) / num_runs, successes / num_runs, samples / (num_runs * 2 ** signal.n)

    def method_report(self, signal, num_runs=10):
        '''
        Reports the results of a method_test.
        '''
        print(
            "Testing SPRIGHT with query method {0}, delays method {1}, reconstruct method {2}."
            .format(self.query_method, self.delays_method, self.reconstruct_method)
        )
        t, s, sam = self.method_test(signal, num_runs)
        print("Average time in seconds: {}".format(t))
        print("Success ratio: {}".format(s))
        print("Average sample ratio: {}".format(sam))


if __name__ == "__main__":
    np.random.seed(10)

    from inputsignal import Signal
    from rna_transform.input_rna_signal import SignalRNA
    from rna_transform.rna_utils import get_rna_base_seq
    q = 4
    n = 10
    N = q ** n
    num_nonzero_indices = 100
    nonzero_indices = np.random.choice(N, num_nonzero_indices, replace=False)
    nonzero_values = 2 + np.random.rand(num_nonzero_indices)
    nonzero_values = nonzero_values * (2 * np.random.binomial(1, 0.5, size=num_nonzero_indices) - 1)
    noise_sd = 1e-6
    positions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    test_signal = SignalRNA(n=n, q=q, noise_sd=800/(q ** n), base_seq=get_rna_base_seq(), positions=positions)
    #test_signal = Signal(n=n, q=q, loc=nonzero_indices, strengths=nonzero_values, noise_sd=noise_sd)
    print("test signal generated")

    spright = QSPRIGHT(
        query_method="complex",
        delays_method="nso",
        reconstruct_method="nso",
        b=6,
        num_subsample=3,
        num_random_delays=20)

    gwht, n_used, peeled = spright.transform(test_signal, verbosity=0, report=True)

    print("found non-zero indices: ")
    print(np.sort(peeled))

    print("true non-zero indices: ")
    print(np.sort(nonzero_indices))
