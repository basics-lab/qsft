import numpy as np
from src.qspright.input_signal import Signal
from src.qspright.utils import lasso_decode


def compute_point_qspright(sparsity, spright, config):
    q = config["q"]
    n = config["n"]
    a = config["a"]
    b = config["b"]
    noise_sd = config["noise_sd"]
    n_points = config["n_points"]
    n_fail = 0
    n_samples = 0
    for j in range(n_points):
        signal = Signal(n=n, q=q, sparsity=sparsity, a=a, b=b, noise_sd=noise_sd)
        gwht, (n_used, n_used_unique, _), peeled = spright.transform(signal, verbose=False, report=True)
        n_fail += (np.sum(np.abs(signal._signal_w - gwht) ** 2) / np.sum(np.abs(signal._signal_w) ** 2)) > 0.1
        n_samples += n_used_unique
    return n_fail/n_points, n_samples/(n_points*(q ** n))


def compute_point_lasso(sparsity, sample_rate, config):
    q = config["q"]
    n = config["n"]
    a = config["a"]
    b = config["b"]
    noise_sd = config["noise_sd"]
    n_points = config["n_points"]
    n_fail = 0
    for j in range(n_points):
        signal = Signal(n=n, q=q, sparsity=sparsity, a=a, b=b, noise_sd=noise_sd)
        gwht, peeled = lasso_decode(signal, sample_rate)
        n_fail += (np.sum(np.abs(signal._signal_w - gwht) ** 2) / np.sum(np.abs(signal._signal_w) ** 2)) > 0.1
    return n_fail/n_points
