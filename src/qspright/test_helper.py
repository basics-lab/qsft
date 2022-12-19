import numpy as np
from src.qspright.lasso import lasso_decode
from src.qspright.qspright import QSPRIGHT
from src.qspright.utils import gwht, dec_to_qary_vec, NpEncoder
import json


class TestHelper:

    def __init__(self, signal_args, methods, subsampling_args, test_args, exp_dir, subsampling=False):

        self.n = signal_args["n"]
        self.q = signal_args["q"]

        self.exp_dir = exp_dir
        self.subsampling = subsampling

        config_path = self.exp_dir / "config.json"
        config_exists = config_path.is_file()

        if not config_exists:
            config_dict = {"query_args": subsampling_args}
            with open(config_path, "w") as f:
                json.dump(config_dict, f, cls=NpEncoder)

        self.signal_args = signal_args
        self.subsampling_args = subsampling_args
        self.test_args = test_args

        if self.subsampling:
            self.train_signal = self.load_train_data()
            # print("Quaternary Training data loaded.", flush=True)
            if any([m.startswith("binary") for m in methods]):
                self.train_signal_binary = self.load_train_data_binary()
                # print("Binary Training data loaded.", flush=True)
            if any([m.startswith("uniform") for m in methods]):
                self.train_signal_uniform = self.load_train_data_uniform()
                # print("Uniform Training data loaded.", flush=True)
            self.test_signal = self.load_test_data()
            # print("Test data loaded.", flush=True)
        else:
            self.train_signal = self.load_full_data()
            self.test_signal = self.train_signal
            if any([m.startswith("binary") for m in methods]):
                raise NotImplementedError  # TODO: implement the conversion
            # print("Full data loaded.", flush=True)

    def generate_signal(self, signal_args):
        raise NotImplementedError

    def load_train_data(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        query_args.update({
            "subsampling_method": "qspright",
            "query_method": "complex",
            "delays_method_source": "identity",
            "delays_method_channel": "nso"
        })
        signal_args["folder"] = self.exp_dir / "train"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)

    def load_train_data_coded(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        query_args.update({
            "subsampling_method": "qspright",
            "query_method": "complex",
            "delays_method_source": "coded",
            "delays_method_channel": "nso"
        })
        signal_args["folder"] = self.exp_dir / "train"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)

    def load_train_data_binary(self):
        return None
    #     signal_args = self.signal_args.copy()
    #     query_args = signal_args["query_args"]
    #     signal_args["n_orig"] = signal_args["n"]
    #     signal_args["q_orig"] = signal_args["q"]
    #     factor = round(np.log(signal_args["q"]) / np.log(2))
    #     signal_args["n"] = factor * signal_args["n"]
    #     signal_args["q"] = 2
    #     signal_args["query_args"]["subsampling_method"] = "qspright"
    #     signal_args["query_args"]["b"] = factor * query_args["b"]
    #     signal_args["query_args"]["all_bs"] = [factor * b for b in query_args["all_bs"]]
    #     signal_args["query_args"]["num_repeat"] = max(1, query_args["num_repeat"] // factor)
    #     signal_args["folder"] = self.exp_dir / "train_binary"
    #     return self.generate_signal(signal_args)

    def load_train_data_uniform(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        n_samples = query_args["num_subsample"] * (signal_args["q"] ** query_args["b"]) *\
                    query_args["num_repeat"] * (signal_args["n"] + 1)
        query_args = {"subsampling_method": "uniform", "n_samples": n_samples}
        signal_args["folder"] = self.exp_dir / "train_uniform"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)

    def load_test_data(self):
        signal_args = self.signal_args.copy()
        (self.exp_dir / "test").mkdir(exist_ok=True)
        signal_args["query_args"] = {"subsampling_method": "uniform", "n_samples": self.test_args.get("n_samples")}
        signal_args["folder"] = self.exp_dir / "test"
        signal_args["noise_sd"] = 0
        return self.generate_signal(signal_args)

    def load_full_data(self):
        #   TODO: implement
        return None

    def compute_model(self, method, model_kwargs, report=False, verbosity=0):
        if method == "gwht":
            return self._calculate_gwht(model_kwargs, report, verbosity)
        elif method == "qspright":
            return self._calculate_qspright(model_kwargs, report, verbosity)
        elif method == "binary_qspright":
            return self._calculate_binary_qspright(model_kwargs, report, verbosity)
        elif method == "uniform_lasso":
            return self._calculate_lasso(model_kwargs, report, verbosity)
        else:
            raise NotImplementedError()

    def test_model(self, method, **kwargs):
        if method == "qspright":
            return self._test_qary(**kwargs)
        elif method == "binary_qspright":
            return self._test_binary(**kwargs)
        elif method == "uniform_lasso":
            return self._test_qary(**kwargs)
        else:
            raise NotImplementedError()

    def _calculate_gwht(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        if verbosity >= 1:
            print("Finding all GWHT coefficients")

        beta = gwht(self.train_signal, q=4, n=self.n)
        print("Found GWHT coefficients")
        return beta

    def _calculate_qspright(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using QSPRIGHT.
        """
        if verbosity >= 1:
            print("Estimating GWHT coefficients with QSPRIGHT")
        qspright = QSPRIGHT(
            reconstruct_method_source="identity",
            reconstruct_method_channel="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=model_kwargs["num_repeat"],
            b=model_kwargs["b"],
            noise_sd=model_kwargs["noise_sd"]
        )
        out = qspright.transform(self.train_signal, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out

    def _calculate_binary_qspright(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using QSPRIGHT.
        """
        factor = round(np.log(self.q) / np.log(2))

        if verbosity >= 1:
            print("Estimating GWHT coefficients with QSPRIGHT")
        qspright = QSPRIGHT(
            reconstruct_method_source="identity",
            reconstruct_method_channel="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=max(1, model_kwargs["num_repeat"] // factor),
            b=factor * model_kwargs["b"],
            noise_sd=model_kwargs["noise_sd"] / factor
        )
        out = qspright.transform(self.train_signal_binary, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out

    def _calculate_lasso(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using LASSO. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        if verbosity > 0:
            print("Finding Fourier coefficients with LASSO")

        out = lasso_decode(self.train_signal_uniform, model_kwargs["n_samples"])

        if verbosity > 0:
            print("Found Fourier coefficients")

        return out

    def _test_qary(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            test_signal = self.test_signal.signal_t
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            batch_size = 10000

            beta_keys = list(beta.keys())
            beta_values = list(beta.values())

            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = dec_to_qary_vec(sample_idx_dec_batch, self.q, self.n)
                freqs = np.array(sample_idx_batch).T @ np.array(beta_keys).T
                H = np.exp(2j * np.pi * freqs / self.q)
                y_hat.append(H @ np.array(beta_values))

            y_hat = np.concatenate(y_hat)

            return np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
        else:
            return 1

    def _test_binary(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            test_signal = self.test_signal.signal_t
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            batch_size = 10000

            beta_keys = list(beta.keys())
            beta_values = list(beta.values())

            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = dec_to_qary_vec(sample_idx_dec_batch, 2, 2 * self.n)
                freqs = np.array(sample_idx_batch).T @ np.array(beta_keys).T
                H = np.exp(2j * np.pi * freqs / 2)
                y_hat.append(H @ np.array(beta_values))

            # TODO: Write with an if clause
            y_hat = np.abs(np.concatenate(y_hat))

            return np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
        else:
            return 1
