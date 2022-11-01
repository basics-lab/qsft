import galois as gl
import math
from src.qspright.coded_sampling.BCH import BCH
import timeit
import numpy as np

if __name__ == '__main__':
    n = 63
    t_max = 4
    p = math.ceil(math.log2(n))
    nt = (2 ** p) - 1 if n <= (2 ** p) - 1 else (2 ** (p+1)) - 1
    valid = gl.bch_valid_codes(nt)
    for n, k, t in valid:
        if t >= t_max:
            bch = BCH(n, k)
            t = t
            break
    s = math.ceil(math.log2(n))
    p = 2*t*s
    c = gl.GF2.Zeros(bch.n)
    c[5] ^= gl.GF2(1)
    c[10] ^= gl.GF2(1)
    H = bch.H.vector()
    GF = bch.field
    print(GF.ufunc_mode)
    D = gl.GF2.Zeros((p, n))
    for i in range(n):
        for j in range(2*t):
            D[s*j:s*(j+1), i] = H[j, i, :]
    H = bch.H
    syndrome = c.view(GF) @ H.T
    syndrome2 = c @ D.T
    syndrome3 = GF.Vector([syndrome2[s*i:s*(i+1)] for i in range(2*t)])
