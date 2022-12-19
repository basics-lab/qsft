import numpy as np
import random
from tqdm import tqdm
import json
from sklearn.linear_model import Lasso
from multiprocessing import Pool
from pathlib import Path
import RNA

from src.qspright.test_helper import TestHelper
from src.qspright.utils import dec_to_qary_vec, qary_vec_to_dec, NpEncoder
from src.qspright.synthetic_signal import generate_signal_w, SyntheticSubsampledSignal, SyntheticSubsampledBinarySignal

class SyntheticHelper(TestHelper):

    def __init__(self, n, q, noise_sd, sparsity, a_min, a_max,
                 baseline_methods, query_args, test_args, exp_dir, subsampling=False):

        self.exp_dir = exp_dir
        config_path = self.exp_dir / "config.json"
        config_exists = config_path.is_file()

        if config_exists:
            with open(config_path) as f:
                config_dict = json.load(f)
            query_args = config_dict["query_args"]
        else:
            config_dict = {"query_args": query_args}
            with open(config_path, "w") as f:
                json.dump(config_dict, f, cls=NpEncoder)

        self.n = n
        self.q = q

        _, locq, strengths = generate_signal_w(n, q, noise_sd, sparsity, a_min, a_max, full=False)
        self.signal_params = {
            "n": self.n,
            "q": self.q,
            "query_args": query_args,
            "locq": locq,
            "strengths": strengths,
        }

        super().__init__(n, q, baseline_methods, query_args, test_args, exp_dir, subsampling)

    def load_train_data(self):
        signal_params = self.signal_params.copy()
        signal_params["query_args"]["subsampling_method"] = "qspright"
        signal_params["folder"] = self.exp_dir / "train"
        return SyntheticSubsampledSignal(**signal_params)

    def load_train_data_binary(self):
        signal_params = self.signal_params.copy()
        query_args = signal_params["query_args"]
        signal_params["n_orig"] = signal_params["n"]
        signal_params["q_orig"] = signal_params["q"]
        factor = round(np.log(signal_params["q"]) / np.log(2))
        signal_params["n"] = factor * signal_params["n"]
        signal_params["q"] = 2
        signal_params["query_args"]["subsampling_method"] = "qspright"
        signal_params["query_args"]["b"] = factor * query_args["b"]
        signal_params["query_args"]["all_bs"] = [factor * b for b in query_args["all_bs"]]
        signal_params["query_args"]["num_random_delays"] = max(1, query_args["num_random_delays"] // factor)
        signal_params["folder"] = self.exp_dir / "train_binary"
        return SyntheticSubsampledBinarySignal(**signal_params)

    def load_train_data_uniform(self):
        signal_params = self.signal_params.copy()
        qa = signal_params["query_args"]
        n_samples = qa["num_subsample"] * (signal_params["q"] ** qa["b"]) *\
                    qa["num_random_delays"] * (signal_params["n"] + 1)
        signal_params["query_args"] = {"subsampling_method": "uniform", "n_samples": n_samples}
        signal_params["folder"] = self.exp_dir / "train_uniform"
        return SyntheticSubsampledSignal(**signal_params)

    def load_test_data(self, test_args=None):
        signal_params = self.signal_params.copy()
        (self.exp_dir / "test").mkdir(exist_ok=True)
        signal_params["query_args"] = {"subsampling_method": "uniform", "n_samples": test_args.get("n_samples")}
        signal_params["folder"] = self.exp_dir / "test"
        signal_params["noise_sd"] = 0
        return SyntheticSubsampledSignal(**signal_params)
