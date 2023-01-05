import numpy as np
from multiprocessing import Pool
import sys
from archive.qspright_nso import QSPRIGHT
from itertools import repeat
from compute_point import compute_point_qspright
from compute_point import compute_point_lasso
import matplotlib.pyplot as plt
sys.path.append("../src/qspright/")
np.random.seed(10)

if __name__ == '__main__':
    spright = QSPRIGHT(
        query_method="simple",
        delays_method="identity",
        reconstruct_method="noiseless",
        num_subsample=2,
        b=3
    )
    config = {
        "q": 3,
        "n": 6,
        "a": 1,
        "b": 10,
        "noise_sd": 0,
        "n_points": 300
    }
    sparsities = [1 + 4*i for i in range(15)]
    p = Pool()
    result = p.starmap(compute_point_qspright, zip(sparsities, repeat(spright), repeat(config)))
    result = np.array(result)
    lasso_result = p.starmap(compute_point_lasso, zip(sparsities, repeat(0.3), repeat(config)))
    p.close()
    plt.plot(sparsities, result[:, 0])
    plt.plot(sparsities, lasso_result)
    plt.xlabel("Sparsity Level")
    plt.ylabel("Fraction of Failures")
    plt.show()
