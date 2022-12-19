import numpy as np
from src.qspright.lasso import lasso_decode
from src.qspright.qspright import QSPRIGHT
from src.qspright.utils import gwht, dec_to_qary_vec


class TestHelper:

    def __init__(self, n, q, baseline_methods, query_args, test_args, exp_dir, subsampling=False):

        self.n = n
        self.q = q

        self.exp_dir = exp_dir
        self.subsampling = subsampling

        print("Sampling Query: ", query_args)
        print("Baseline Methods: ", baseline_methods)

        if self.subsampling:
            self.train_signal = self.load_train_data()
            print("Quaternary Training data loaded.", flush=True)
            if any([m.startswith("binary") for m in baseline_methods]):
                self.train_signal_binary = self.load_train_data_binary()
                print("Binary Training data loaded.", flush=True)
            if any([m.startswith("uniform") for m in baseline_methods]):
                self.train_signal_uniform = self.load_train_data_uniform()
                print("Uniform Training data loaded.", flush=True)
            self.test_signal = self.load_test_data(test_args)
            print("Test data loaded.", flush=True)
        else:
            self.train_signal = self.load_full_data()
            self.test_signal = self.train_signal
            if any([m.startswith("binary") for m in baseline_methods]):
                raise NotImplementedError  # TODO: implement the conversion
            print("Full data loaded.", flush=True)

    def load_train_data(self):
        raise NotImplementedError()

    def load_train_data_binary(self):
        raise NotImplementedError()

    def load_train_data_uniform(self):
        raise NotImplementedError()

    def load_full_data(self):
        raise NotImplementedError()

    def load_test_data(self, test_args=None):
        raise NotImplementedError()

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
            reconstruct_method="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_random_delays=model_kwargs["num_random_delays"],
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
            reconstruct_method="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_random_delays=max(1, model_kwargs["num_random_delays"] // factor),
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
