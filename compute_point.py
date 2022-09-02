import numpy as np
from src.qspright.inputsignal import Signal


def compute_point(sparsity, spright):
    q = 4
    n = 10
    a = 1
    b = 10
    noise_sd = 0
    n_fail = 0
    n_points = 100
    for j in range(n_points):
        signal = Signal(n=n, q=q, sparsity=sparsity, a=a, b=b, noise_sd=noise_sd)
        gwht, (n_used, n_used_unique), peeled = spright.transform(signal, verbose=False, report=True)
        n_fail += (np.sum(np.abs(signal._signal_w - gwht) ** 2) / np.sum(np.abs(signal._signal_w) ** 2)) > 0.1
    return n_fail/n_points
