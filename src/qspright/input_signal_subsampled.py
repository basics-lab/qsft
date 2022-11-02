from src.qspright.input_signal import Signal
from src.qspright.query import get_Ms_and_Ds
from src.qspright.utils import qary_ints, qary_vec_to_dec, dec_to_qary_vec, load_data, save_data
import numpy as np
from pathlib import Path
import time
import random
from src.qspright.query import compute_delayed_gwht
import galois as gl

class SubsampledSignal(Signal):
    CANNOT_TRANSFORM = 10

    def _set_params(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.N = self.q ** self.n
        self.signal_t = kwargs.get("signal_t")
        self.signal_w = kwargs.get("signal_w")
        self.query_args = kwargs.get("query_args")
        self.b = self.query_args.get("b")
        self.all_bs = self.query_args.get("all_bs")
        self.num_subsample = self.query_args.get("num_subsample")
        self.num_random_delays = self.query_args.get("num_repeat", 1)
        self.subsampling_method = self.query_args.get("subsampling_method")
        self.delays_method_source = self.query_args.get("delays_method_source")

        self.L = None  # List of all length b qary vectors
        self.foldername = kwargs.get("folder")

    def _init_signal(self):
        start_time = time.time()

        if self.subsampling_method == "qspright":
            self._set_Ms_and_Ds_qspright()
            transforms_exist = self._check_transforms_qspright()
        else:
            transforms_exist = False

        if self.signal_t is None:
            self.signal_t = {}
            if self.subsampling_method == "qspright" and not transforms_exist:
                self._subsample_qspright()
            elif self.subsampling_method == "uniform":
                self._subsample_uniform()

        end_time = time.time()
        print(f"Sample generation/load time: {end_time - start_time} s")

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
        if self.foldername:
            Path(f"{self.foldername}/samples").mkdir(exist_ok=True)
            for i in range(len(self.Ms)):
                sample_file = Path(f"{self.foldername}/samples/M{i}.pickle")
                if sample_file.is_file():
                    signal_t = load_data(sample_file)
                else:
                    query_indices = self._get_qspright_query_indices(self.Ms[i], self.Ds[i])
                    signal_t = self.subsample(query_indices)
                    save_data(signal_t, sample_file)
                self.signal_t.update(signal_t)
        else:
            for i in range(len(self.Ms)):
                query_indices = self._get_qspright_query_indices(self.Ms[i], self.Ds[i])
                signal_t = self.subsample(query_indices)
                self.signal_t.update(signal_t)

    def _subsample_uniform(self):
        if self.foldername:
            Path(f"{self.foldername}").mkdir(exist_ok=True)
            sample_file = Path(f"{self.foldername}/signal_t.pickle")
            if sample_file.is_file():
                signal_t = load_data(sample_file)
            else:
                query_indices = self._get_random_query_indices(self.query_args["n_samples"])
                signal_t = self.subsample(query_indices)
                save_data(signal_t, sample_file)
            self.signal_t.update(signal_t)
        else:
            query_indices = self._get_random_query_indices(self.query_args["n_samples"])
            signal_t = self.subsample(query_indices)
            self.signal_t.update(signal_t)

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

    def _get_qspright_query_indices(self, M, D):
        L = self.get_all_qary_vectors()
        ML = (M @ L) % self.q
        num_subdelays = len(D[0])
        base_inds = [[(ML.T + D[k][j]) % self.q for j in range(num_subdelays)] for k in
                     range(self.num_random_delays)]
        base_inds = np.array(base_inds)
        base_inds = np.reshape(base_inds, (-1, base_inds.shape[-1]))
        base_inds_dec = qary_vec_to_dec(base_inds.T, self.q).T
        return list(zip(base_inds, base_inds_dec))

    def _get_random_query_indices(self, n_samples):
        n_samples = np.minimum(n_samples, self.N)
        base_inds_dec = random.sample(range(self.N), n_samples)
        base_inds = np.array(dec_to_qary_vec(base_inds_dec, self.q, self.n)).T
        return list(zip(base_inds, base_inds_dec))

    def get_MDUS(self, ret_num_subsample, ret_num_random_delays, b):
        Ms_ret = []
        Ds_ret = []
        Us_ret = []
        Ss_ret = []
        if ret_num_subsample <= self.num_subsample and ret_num_random_delays <= self.num_random_delays and b <= self.b:
            subsample_idx = np.random.choice(self.num_subsample, ret_num_subsample, replace=False)
            delay_idx = np.random.choice(self.num_random_delays, ret_num_random_delays, replace=False)
            for i in subsample_idx:
                Ms_ret.append(self.Ms[i][:, (self.b - b):])
                Ds_ret.append([])
                Us_ret.append([])
                Ss_ret.append([])
                for j in delay_idx:
                    Ds_ret[-1].append(self.Ds[i][j])
                    Us_ret[-1].append(self.Us[b][i][j])
                    Ss_ret[-1].append(self.used_samples[b][i][j])
            return Ms_ret, Ds_ret, Us_ret, Ss_ret
        else:
            raise ValueError("There are not enough Ms or Ds.")

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
