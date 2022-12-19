from src.qspright.input_signal import Signal
from src.qspright.query import get_Ms_and_Ds, compute_delayed_gwht
from src.qspright.utils import qary_ints, qary_vec_to_dec, dec_to_qary_vec, gwht, load_data, save_data
import numpy as np
from pathlib import Path
import time
import random
<<<<<<< .merge_file_qFtufq
from tqdm import tqdm

=======
from src.qspright.query import compute_delayed_gwht
import galois as gl
>>>>>>> .merge_file_j5gClo

class SubsampledSignal(Signal):

    def _set_params(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.N = self.q ** self.n
        self.signal_w = kwargs.get("signal_w")
        self.query_args = kwargs.get("query_args")
        self.b = self.query_args.get("b")
        self.all_bs = self.query_args.get("all_bs")
        self.num_subsample = self.query_args.get("num_subsample")
        self.num_random_delays = self.query_args.get("num_repeat", 1)
        self.subsampling_method = self.query_args.get("subsampling_method")
<<<<<<< .merge_file_qFtufq
        self.samples_t = None
=======
        self.delays_method_source = self.query_args.get("delays_method_source")

>>>>>>> .merge_file_j5gClo
        self.L = None  # List of all length b qary vectors
        self.foldername = kwargs.get("folder")

    def _init_signal(self):
        start_time = time.time()

        if self.subsampling_method == "qspright":
            self._set_Ms_and_Ds_qspright()
            self._subsample_qspright()

        if self.subsampling_method == "uniform":
            self._subsample_uniform()

        end_time = time.time()
        print(f"Sample generation/load time: {end_time - start_time} s")

<<<<<<< .merge_file_qFtufq
=======
        if self.subsampling_method == "qspright" and self.all_bs:
            start_time = time.time()
            print("Computing/loading sub-transforms...", flush=True)
            self._transform_qspright()
            end_time = time.time()
            print(f"Sub-transform generation/load time: {end_time - start_time} s")
        elif self.subsampling_method == "qspright":
            start_time = time.time()
            print("Computing/loading transform...", flush=True)
            self._transform_qspright()
            end_time = time.time()
            print(f"Transform generation/load time: {end_time - start_time} s")

    def _check_transforms_qspright(self):
        if self.foldername:
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)
            for b in self.all_bs:
                for i in range(len(self.Ms)):
                    Us_path = Path(f"{self.foldername}/transforms/U{i}_b{b}.pickle")
                    if not Us_path.is_file():
                        return False
            return True
        else:
            return False

    def _transform_qspright(self):
        self.Us = {}
        self.used_samples = {}
        if self.foldername:
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)
            for b in self.all_bs:
                U_b = []
                used_b = []
                for i in range(len(self.Ms)):
                    Us_path = Path(f"{self.foldername}/transforms/U{i}_b{b}.pickle")
                    if Us_path.is_file():
                        U_ib, used_ib = load_data(Us_path)
                    else:
                        U_ib, used_ib = self._calc_transforms(self.Ms[i], self.Ds[i], b)
                        save_data((U_ib, used_ib), Us_path)
                    U_b.append(U_ib)
                    used_b.append(used_ib)
                self.Us[b] = U_b
                self.used_samples[b] = used_b
        else:
            U_b = []
            used_b = []
            for i in range(len(self.Ms)):
                U_ib, used_ib = self._calc_transforms(self.Ms[i], self.Ds[i], self.b)
                U_b.append(U_ib)
                used_b.append(used_ib)
            self.Us[self.b] = U_b
            self.used_samples[self.b] = used_b



>>>>>>> .merge_file_j5gClo
    def _set_Ms_and_Ds_qspright(self):
        if self.foldername:
            Path(f"{self.foldername}").mkdir(exist_ok=True)
            Ms_and_Ds_path = Path(f"{self.foldername}/Ms_and_Ds.pickle")
            if Ms_and_Ds_path.is_file():
                self.Ms, self.Ds = load_data(Ms_and_Ds_path)
            else:
                self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)
                save_data((self.Ms, self.Ds), f"{self.foldername}/Ms_and_Ds.pickle")
        else:
            self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)

    def _subsample_qspright(self):
        self.Us = []
        self.transformTimes = []

        if self.foldername:
            Path(f"{self.foldername}/samples").mkdir(exist_ok=True)
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)

        pbar = tqdm(total=len(self.Ms) * len(self.Ds[0]) * (self.n + 1), position=0)
        for i in range(len(self.Ms)):
            U_i = []
            T_i = []
            for j in range(len(self.Ds[0])):
                transform_file = Path(f"{self.foldername}/transforms/U{i}_{j}.pickle")
                if self.foldername and transform_file.is_file():
                    U_ij, T_ij = load_data(transform_file)
                    pbar.update(self.n + 1)
                else:
                    U_ij = {}
                    T_ij = {}
                    sample_file = Path(f"{self.foldername}/samples/M{i}_D{j}.pickle")
                    if self.foldername and sample_file.is_file():
                        samples = load_data(sample_file)
                        pbar.update(self.n+1)
                    else:
                        query_indices = self._get_qspright_query_indices(self.Ms[i], self.Ds[i][j])
                        samples = []
                        for k in range(len(query_indices)):
                            samples.append(self.subsample(query_indices[k]))
                            pbar.update()
                        if self.foldername:
                            save_data(samples, sample_file)
                    for b in self.all_bs:
                        start_time = time.time()
                        U_ij[b] = self._compute_subtransform(samples, b)
                        T_ij[b] = time.time() - start_time
                    if self.foldername:
                        save_data((U_ij, T_ij), transform_file)

                U_i.append(U_ij)
                T_i.append(T_ij)

            self.Us.append(U_i)
            self.transformTimes.append(T_i)

    def _subsample_uniform(self):
        if self.foldername:
            Path(f"{self.foldername}").mkdir(exist_ok=True)

        sample_file = Path(f"{self.foldername}/signal_t.pickle")
        if self.foldername and sample_file.is_file():
            signal_t = load_data(sample_file)
        else:
            query_indices = self._get_random_query_indices(self.query_args["n_samples"])
            samples = self.subsample(query_indices)
            signal_t = dict(zip(query_indices, samples))
            if self.foldername:
                save_data(signal_t, sample_file)
        self.signal_t = signal_t

    def get_all_qary_vectors(self):
        if self.L is None:
            self.L = np.array(qary_ints(self.b, self.q))  # List of all length b qary vectors
        return self.L

    def subsample(self, query_indices):
        raise NotImplementedError

    def get_time_domain(self, base_inds, dec=True):
        base_inds = np.array(base_inds)
        if dec:
            if len(base_inds.shape) == 3:
                sample_array = [qary_vec_to_dec(inds, self.q) for inds in base_inds]
                return [np.array([self.signal_t[tup] for tup in inds]) for inds in sample_array]
            elif len(base_inds.shape) == 2:
                sample_array = [tuple(base_inds[:, i]) for i in range(base_inds.shape[1])]
                return np.array([self.signal_t[tup] for tup in sample_array])
        else:
            if len(base_inds.shape) == 3:
                sample_array = [[tuple(inds[:, i]) for i in range(inds.shape[1])] for inds in base_inds]
                return [np.array([self.signal_t[tup] for tup in inds]) for inds in sample_array]
            elif len(base_inds.shape) == 2:
                sample_array = [tuple(base_inds[:, i]) for i in range(base_inds.shape[1])]
                return np.array([self.signal_t[tup] for tup in sample_array])

    def _get_qspright_query_indices(self, M, D_sub):
        b = M.shape[1]
        L = self.get_all_qary_vectors()
        ML = (M @ L) % self.q
        base_inds = [(ML + np.outer(d, np.ones(self.q ** b, dtype=int))) % self.q for d in D_sub]
        base_inds = np.array(base_inds)
        base_inds_dec = []
        for i in range(len(base_inds)):
            base_inds_dec.append(qary_vec_to_dec(base_inds[i], self.q))
        return base_inds_dec

    def _get_random_query_indices(self, n_samples):
        n_samples = np.minimum(n_samples, self.N)
        base_inds_dec = random.sample(range(self.N), n_samples)
        return base_inds_dec

    def get_MDU(self, ret_num_subsample, ret_num_random_delays, b, trans_times=False):
        Ms_ret = []
        Ds_ret = []
        Us_ret = []
        Ts_ret = []
        if ret_num_subsample <= self.num_subsample and ret_num_random_delays <= self.num_random_delays and b <= self.b:
            subsample_idx = np.random.choice(self.num_subsample, ret_num_subsample, replace=False)
            delay_idx = np.random.choice(self.num_random_delays, ret_num_random_delays, replace=False)
            for i in subsample_idx:
                Ms_ret.append(self.Ms[i][:, :b])
                Ds_ret.append([])
                Us_ret.append([])
                Ts_ret.append([])
                for j in delay_idx:
                    Ds_ret[-1].append(self.Ds[i][j])
                    Us_ret[-1].append(self.Us[i][j][b])
                    Ts_ret[-1].append(self.transformTimes[i][j][b])
            if trans_times:
                return Ms_ret, Ds_ret, Us_ret, Ts_ret
            else:
                return Ms_ret, Ds_ret, Us_ret
        else:
            raise ValueError("There are not enough Ms or Ds.")

<<<<<<< .merge_file_qFtufq
    def _compute_subtransform(self, samples, b):
        transform = [gwht(row[::(self.q ** (self.b - b))], self.q, b) for row in samples]
        return transform


=======
    def _calc_transforms(self, M, D, b):
        U = []
        used_samples = []
        for D_sub in D:
            U_sub, used_i = compute_delayed_gwht(self, M[:, (self.b - b):], D_sub, self.q)
            U.append(U_sub)
            used_samples.append(len(used_i))
        return U, used_samples

    def get_source_parity(self):
        return self.Ds[0][0].shape[0]
>>>>>>> .merge_file_j5gClo
