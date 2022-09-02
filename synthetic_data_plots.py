import numpy as np
from multiprocessing import Pool
import sys
from src.qspright.qspright_nso import QSPRIGHT
from itertools import repeat
from compute_point import compute_point
import matplotlib.pyplot as plt
sys.path.append("./src/qspright/")
np.random.seed(10)

if __name__ == '__main__':
    spright = QSPRIGHT(
        query_method="simple",
        delays_method="identity",
        reconstruct_method="noiseless",
        num_subsample=2,
        b=4
    )
    sparsities = [20*(i+1) for i in range(15)]
    p = Pool()
    result = p.starmap(compute_point, zip(sparsities, repeat(spright)))
    plt.plot(sparsities, result)
    plt.xlabel("Sparsity Level")
    plt.ylabel("Fraction of Failures")
    plt.show()
