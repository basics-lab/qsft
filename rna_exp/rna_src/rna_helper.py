import numpy as np
import json
from qsft.test_helper import TestHelper
from qsft.utils import NpEncoder
from rna_exp.rna_src.input_rna_signal_subsampled import RnaSubsampledSignal
from rna_exp.rna_src.rna_utils import get_rna_base_seq


class RNAHelper(TestHelper):
    mfe_base = 0
    base_seq_list = None
    positions = None

    def __init__(self, signal_args, methods, subsampling_args, test_args, exp_dir, subsampling=False):

        config_path = exp_dir / "config.json"
        config_exists = config_path.is_file()

        if config_exists:
            with open(config_path) as f:
                config_dict = json.load(f)
            subsampling_args = config_dict["subsampling_args"]
            signal_args = config_dict["signal_args"]
        else:
            positions = list(np.sort(np.random.choice(len(get_rna_base_seq()), size=signal_args["n"], replace=False)))
            signal_args.update({
                "base_seq": get_rna_base_seq(),
                "positions": positions
            })
            config_dict = {"signal_args": signal_args, "subsampling_args": subsampling_args}
            with open(config_path, "w") as f:
                json.dump(config_dict, f, cls=NpEncoder)

        print("Positions: ", signal_args["positions"])
        super().__init__(signal_args, methods, subsampling_args, test_args, exp_dir, subsampling)

    def generate_signal(self, signal_args):
        return RnaSubsampledSignal(**signal_args)
