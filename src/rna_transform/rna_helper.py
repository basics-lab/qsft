import numpy as np
import random
from tqdm import tqdm
import json
from sklearn.linear_model import Lasso
from multiprocessing import Pool
from pathlib import Path
import RNA

from src.qspright.test_helper import TestHelper
from src.qspright.utils import dec_to_qary_vec,qary_vec_to_dec, NpEncoder
from src.rna_transform.input_rna_signal import RnaSignal
from src.rna_transform.input_rna_signal_subsampled import RnaSubsampledSignal
from src.rna_transform.rna_utils import get_rna_base_seq


class RNAHelper(TestHelper):
    mfe_base = 0
    base_seq_list = None
    positions = None

    @staticmethod
    def _calc_data_inst_q4(query_indices):
        full = next_q4(RNAHelper.base_seq_list, RNAHelper.positions, query_indices)
        fc = RNA.fold_compound(full)
        (_, mfe) = fc.mfe()
        return mfe - RNAHelper.mfe_base

    @staticmethod
    def _calc_data_inst_q2(query_indices):
        full = next_q2(RNAHelper.base_seq_list, RNAHelper.positions, query_indices)
        fc = RNA.fold_compound(full)
        (_, mfe) = fc.mfe()
        return mfe - RNAHelper.mfe_base

    def __init__(self, n, baseline_methods, query_args, test_args, exp_dir, subsampling=False):

        self.exp_dir = exp_dir
        config_path = self.exp_dir / "config.json"
        config_exists = config_path.is_file()

        if config_exists:
            with open(config_path) as f:
                config_dict = json.load(f)
            query_args = config_dict["query_args"]
        else:
            positions = list(np.sort(np.random.choice(len(get_rna_base_seq()), size=n, replace=False)))
            config_dict = {"base_seq": get_rna_base_seq(), "positions": positions, "query_args": query_args}
            with open(config_path, "w") as f:
                json.dump(config_dict, f, cls=NpEncoder)

        self.positions = config_dict["positions"]
        self.base_seq = config_dict["base_seq"]
        self.n = len(self.positions)

        (_, mfe_base) = RNA.fold("".join(self.base_seq))
        RNAHelper.base_seq_list = np.array(list(self.base_seq))
        RNAHelper.positions = self.positions
        RNAHelper.mfe_base = mfe_base

        self.query_args = query_args

        print("Positions: ", self.positions)
        super().__init__(n, q, baseline_methods, query_args, test_args, exp_dir, subsampling)

    def load_train_data(self):
        query_args = self.query_args.copy()
        query_args["subsampling_method"] = "qspright"
        return RnaSubsampledSignal(q=4,
                                   query_args=query_args, base_seq=self.base_seq, positions=self.positions,
                                   sampling_function=RNAHelper._calc_data_inst_q4, folder=self.exp_dir / "train")

    def load_train_data_binary(self):
        query_args = self.query_args.copy()
        query_args["subsampling_method"] = "qspright"
        query_args["b"] = 2 * query_args["b"]
        query_args["all_bs"] = [2 * b for b in query_args["all_bs"]]
        query_args["num_repeat"] = query_args["num_repeat"] // 2
        return RnaSubsampledSignal(q=2,
                                   query_args=query_args, base_seq=self.base_seq, positions=self.positions,
                                   sampling_function=RNAHelper._calc_data_inst_q2, folder=self.exp_dir / "train_binary")

    def load_full_data(self):
        return RnaSignal(q=4, positions=self.positions, base_seq=self.base_seq,
                         sampling_function=self._calc_data_inst_q4, folder=self.exp_dir / "full")

    def load_train_data_uniform(self):
        signal_params = self.signal_params.copy()
        qa = signal_params["query_args"]
        n_samples = qa["num_subsample"] * (qa["q"] ** qa["b"]) * qa["num_repeat"] * (qa["n"] + 1)
        signal_params["query_args"] = {"subsampling_method": "uniform", "n_samples": n_samples}
        signal_params["folder"] = self.exp_dir / "train_uniform"
        return RnaSubsampledSignal(q=2,
                                   query_args=query_args, base_seq=self.base_seq, positions=self.positions,
                                   sampling_function=RNAHelper._calc_data_inst_q2, folder=self.exp_dir / "train_binary")

    def load_test_data(self, test_args=None):
        (self.exp_dir / "test").mkdir(exist_ok=True)
        query_args = {"subsampling_method": "uniform", "n_samples": test_args.get("n_samples")}
        return RnaSubsampledSignal(n=self.n, q=4, query_args=query_args,
                                   base_seq=self.base_seq, positions=self.positions,
                                   sampling_function=self._calc_data_inst_q4, folder=self.exp_dir / "test")

nucs = np.array(["A", "U", "C", "G"])
q = 4
def next_q4(base_seq_list, positions, query_index):
    seq = nucs[query_index]
    base_seq_list[positions] = seq
    return "".join(base_seq_list)

def next_q2(base_seq_list, positions, query_index):
    idx = dec_to_qary_vec([qary_vec_to_dec(query_index, 2)], 4, len(query_index)//2)[:, 0]
    seq = nucs[idx]
    base_seq_list[positions] = seq
    return "".join(base_seq_list)
