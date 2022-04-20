import time
import numpy as np
import tqdm
import sys
sys.path.append("src")
from src.utils import qary_ints,  bin_to_dec, qary_vec_to_dec, nth_roots_unity, dec_to_qary_vec, near_nth_roots
from src.query import compute_delayed_gwht, get_Ms, get_b, get_D


class QSPRIGHT:
    '''
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
    '''
    def __init__(self, query_method, delays_method, reconstruct_method):
        self.query_method = query_method
        self.delays_method = delays_method
        self.reconstruct_method = reconstruct_method

    def transform(self, signal, verbose=False, report=False):
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
        omega = np.exp(2j * np.pi / q)
        result = []
        gwht = np.zeros_like(signal.signal_t)
        b = get_b(signal, method=self.query_method)
        peeling_max = q ** b
        Ms = get_Ms(signal.n, b, q, method=self.query_method)
        Ms = [np.array(M) for M in Ms]
        Us, Ss = [], []
        eps = 1e-7
        singletons = {}
        multitons = []
        if report:
            used = set()
        if self.delays_method != "nso":
            num_delays = signal.n + 1
        else:
            num_delays = signal.n * int(np.log2(signal.n)) # idk
        K = np.array(qary_ints(signal.n, q))

        used = set()
        peeled = set([])
        # subsample, make the observation [U] and offset signature [S] matrices
        for M in Ms:
            D = get_D(signal.n, method=self.delays_method, num_delays=num_delays, q=signal.q)
            D = np.array(D)
            if verbose:
                print("------")
                print("a delay matrix")
                print(D)
            U, used_i = compute_delayed_gwht(signal, M, D, q)
            Us.append(U)
            Ss.append(omega ** np.array(D @ K)) # offset signature matrix
            if report:
                used = used.union(used_i)

        cutoff = 2 * signal.noise_sd ** 2 * (2 ** (signal.n - b)) * num_delays # noise threshold
        if verbose:
            print('cutoff: {}'.format(cutoff))
        # K is the binary representation of all integers from 0 to 2 ** n - 1.
        select_froms = [(M.T @ K) % q for M in Ms]
        # `select_froms` is the collection of 'j' values and associated indices
        # so that we can quickly choose from the coefficient locations such that M.T @ k = j as in (20)
        # example: ball j goes to bin at "select_froms[i][j]"" in stage i

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
        max_iter= 3
        iter_step = 0
        there_were_multitons = True
        while there_were_multitons and num_peeling < peeling_max and iter_step < max_iter:
            iter_step += 1
            if verbose:
                print('-----')
                print('the measurement matrix')
                for U in Us:
                    print(U)
            # first step: find all the singletons and multitons.
            singletons = {}  # dictionary from (i, j) values to the true index of the singleton, k.
            multitons = []  # list of (i, j) values indicating where multitons are.
            for i, (U, S, select_from) in enumerate(zip(Us, Ss, select_froms)):
                for j, col in enumerate(U.T):
                    if verbose:
                        j_qary = np.array(dec_to_qary_vec(np.array([j]), q, b))[:, 0]
                        active_k_idx = []
                        for idx in range(select_from.shape[1]):
                            if np.all(j_qary == select_from[:, idx]):
                                active_k_idx.append(idx)
                        k_active = K[:, active_k_idx]
                        print("For M(" + str(i) + ") entry U(" + str(j) + ") the active indicies are:")
                        print(active_k_idx)
                        print("The active and non-zero (unpeeled) indicies are:")
                        non_zeros = set(np.nonzero(signal.signal_w > 0)[0])
                        active_non_zero = list(set(active_k_idx).intersection(non_zeros).difference(peeled))
                        print(active_non_zero)
                    if np.vdot(col, col) > eps:
                        ratios = col[0] / col
                        is_singleton = near_nth_roots(ratios, q, eps)
                        if is_singleton:
                            singleton_ind = (
                                (np.arange(q) @ (np.abs(ratios - np.outer(nth_roots_unity(q), np.ones(signal.n + 1)))
                                                 < eps))[1:])
                            dec_singleton_ind = qary_vec_to_dec(singleton_ind, q)
                            rho = np.vdot(S[:, dec_singleton_ind], col) / len(col)
                            singletons[(i, j)] = (singleton_ind, rho)
                            if verbose:
                                print("We predict that the singleton index is " + str(dec_singleton_ind))
                        else:
                            multitons.append((i, j))
                            if verbose:
                                print("We have a Multiton")
                    else:
                        if verbose:
                            print("We have a zeroton!")

            # all singletons and multitons are discovered
            if verbose:
                print('singletons:')
                for ston in singletons.items():
                    print("\t{0} {1}\n".format(ston, bin_to_dec(ston[1][0])))

                print("Multitons : {0}\n".format(multitons))

            # raise RuntimeError("stop")
            # WARNING: this is not a correct thing to do
            # in the last iteration of peeling, everything will be singletons and there
            # will be no multitons
            if len(multitons) == 0: # no more multitons, and can construct final WHT
                there_were_multitons = False

            # balls to peel
            balls_to_peel = set()
            ball_values = {}
            for (i, j) in singletons:
                k, rho = singletons[(i, j)]
                ball = qary_vec_to_dec(k, q)
                balls_to_peel.add(ball)
                ball_values[ball] = rho
                result.append((k, ball_values[ball]))

            if verbose:
                print('these balls will be peeled')
                print(balls_to_peel)
            # peel
            for ball in balls_to_peel:
                num_peeling += 1
                peeled.add(ball)
                k = np.array(dec_to_qary_vec(np.array([ball]), q, signal.n))
                potential_peels = [(l, qary_vec_to_dec(M.T.dot(k) % q, q)) for l, M in enumerate(Ms)]
                if verbose:
                    print("Processing Singleton {0}".format(ball))
                    print(k)
                    for (l, j) in potential_peels:
                        print("The singleton appears in M({0}), U({1})".format(l, j))
                for peel in potential_peels:
                    signature_in_stage = Ss[peel[0]][:,ball]
                    to_subtract = ball_values[ball] * signature_in_stage.reshape(-1, 1)
                    if verbose:
                        print('this is subtracted:')
                        print(to_subtract)
                        print('from')
                        print(Us[peel[0]][:, peel[1]])
                        print("Peeled ball {0} off bin {1}".format(qary_vec_to_dec(k, q), peel))
                    Us[peel[0]][:, peel[1]] -= to_subtract
                if verbose:
                    print("Iteration Complete: The peeled indicies are:")
                    print(peeled)
        loc = set()
        for k, value in result: # iterating over (i, j)s
            idx = qary_vec_to_dec(k, q) # converting 'k's of singletons to decimals
            loc.add(idx)
            if gwht[idx] == 0:
                gwht[idx] = value
            else:
                gwht[idx] = (gwht[idx] + value) / 2
                # average out noise; e.g. in the example in 3.2, U1[11] and U2[11] are the same singleton,
                # so averaging them reduces the effect of noise.
        if not report:
            return gwht
        else:
            return gwht, len(used), loc

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
    from src.inputsignal import Signal
    test_signal = Signal(3, [4, 6, 10, 15, 24, 37, 48, 54], q=4, strengths=[2, 4, 1, 1, 1, 3, 8, 1], noise_sd=0)
    spright = QSPRIGHT(
        query_method="complex",
        delays_method="complex",
        reconstruct_method="noiseless"
    )
    spright_tf = spright.transform(test_signal, verbose=True)
