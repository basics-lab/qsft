import time
import numpy as np
from src.qspright.reconstruct import singleton_detection
from src.qspright.utils import bin_to_dec, qary_vec_to_dec, sort_qary_vecs, calc_hamming_weight, dec_to_qary_vec
from src.qspright.synthetic_signal import SyntheticSubsampledSignal


class QSPRIGHT:

    def __init__(self, **kwargs):
        self.reconstruct_method_source = kwargs.get("reconstruct_method_source")
        self.reconstruct_method_channel = kwargs.get("reconstruct_method_channel")
        self.num_subsample = kwargs.get("num_subsample")
        self.num_repeat = kwargs.get("num_repeat")
        self.b = kwargs.get("b")
        self.noise_sd = kwargs.get("noise_sd")
        self.source_decoder = kwargs.get("source_decoder", None)

    def transform(self, signal, verbosity=0, report=False, timing_verbose=False, **kwargs):

        q = signal.q
        n = signal.n
        b = self.b

        omega = np.exp(2j * np.pi / q)
        result = []

        gwht = {}
        gwht_counts = {}

        peeling_max = q ** n
        peeled = set([])

        Ms, Ds, Us, Ts = signal.get_MDU(self.num_subsample, self.num_repeat, self.b, trans_times=True)

        for i in range(len(Ds)):
            Us[i] = np.vstack(Us[i])
            Ds[i] = np.vstack(Ds[i])

        transform_time = np.sum(Ts)

        peeling_start = time.time()

        Us = np.array(Us)

        if type(signal) is SyntheticSubsampledSignal:
            Us += np.random.normal(0, self.noise_sd, size=Us.shape + (2,)).view(np.complex).reshape(Us.shape)

        gamma = 1.5

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
        num_peeling = 0

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
            for i, (U, M, D) in enumerate(zip(Us, Ms, Ds)):
                for j, col in enumerate(U.T):
                    if np.linalg.norm(col) ** 2 > cutoff * len(col):
                        k = singleton_detection(
                            col,
                            method_channel=self.reconstruct_method_channel,
                            method_source=self.reconstruct_method_source,
                            q=q,
                            source_parity=signal.get_source_parity(),
                            nso_subtype="nso1",
                            source_decoder=self.source_decoder
                        )
                        signature = omega ** (D @ k)
                        rho = np.dot(np.conjugate(signature), col) / D.shape[0]
                        residual = col - rho * signature

                        j_qary = dec_to_qary_vec([j], q, b).T[0]
                        bin_matching = np.all((M.T @ k) % q == j_qary)

                        if verbosity >= 5:
                            print((i, j), np.linalg.norm(residual) ** 2, cutoff * len(residual))
                        if (not bin_matching) or np.linalg.norm(residual) ** 2 > cutoff * len(residual):
                            multitons.append((i, j))
                            if verbosity >= 6:
                                print("We have a Multiton")
                        else:  # declare as singleton
                            singletons[(i, j)] = (k, rho)
                            if verbosity >= 3:
                                print("We have a Singleton at " + str(k))
                    else:
                        if verbosity >= 6:
                            print("We have a Zeroton")

            # all singletons and multitons are discovered
            if verbosity >= 5:
                print('singletons:')
                for ston in singletons.items():
                    print("\t{0} {1}\n".format(ston, bin_to_dec(ston[1][0])))

                print("Multitons : {0}\n".format(multitons))

            # if there were no multi-tons or single-tons, decrease cutoff
            if len(multitons) == 0 or len(singletons) == 0:
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
                    # print(np.linalg.norm(Us[peel[0]][:, peel[1]]), np.linalg.norm(to_subtract))
                    if verbosity >= 6:
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

        peeling_time = time.time() - peeling_start

        if not report:
            return gwht
        else:
            n_samples = np.prod(np.shape(np.array(Us)))
            if len(loc) > 0:
                loc = list(loc)
                if kwargs.get("sort", False):
                    loc = sort_qary_vecs(loc)
                avg_hamming_weight = np.mean(calc_hamming_weight(loc))
                max_hamming_weight = np.max(calc_hamming_weight(loc))
            else:
                loc, avg_hamming_weight, max_hamming_weight = [], 0, 0
            result = {
                "gwht": gwht,
                "runtime": transform_time + peeling_time,
                "n_samples": n_samples,
                "locations": loc,
                "avg_hamming_weight": avg_hamming_weight,
                "max_hamming_weight": max_hamming_weight
            }
            return result