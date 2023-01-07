# Efficient Sparse Fourier Transforms

This repository contains code for the paper:

_"Efficiently Computing Sparse Fourier Transforms of $q$-ary Functions" Yigit Erginbas*, Justin Kang*, Amirali Aghazadeh, Kannan Ramchandran_

Fourier transformations of pseudo-Boolean functions are popular tools for analyzing functions of binary sequences. Real-world functions often have structures that manifest in a sparse Fourier transform, and previous works have shown that under the assumption of sparsity the transform can be computed efficiently. But what if we want to compute the Fourier transform of functions defined over a $q$-ary alphabet? These types of functions arise naturally in many areas including biology. A typical workaround is to encode the $q$-ary sequence in binary however, this approach is computationally inefficient and fundamentally incompatible with the existing sparse Fourier transform techniques. Herein, we develop a sparse Fourier transform algorithm specifically for $q$-ary functions of length $n$ sequences, dubbed $q$-SFT, which provably computes an $S$-sparse transform with vanishing error as $q^n$ goes to $\infty$ in $O(Sn)$ function evaluations and $O(S n^2 \log q)$ computations, where $S = q^{n\delta}$ for some $\delta < 1$. Under certain assumptions, we show that for fixed $q$, a robust version of $q$-SFT has a sample complexity of $O(Sn^2)$ and a computational complexity of $O(Sn^3)$ with the same asymptotic guarantees. We present numerical simulations on synthetic and real-world RNA data, demonstrating the scalability of $q$-SFT to massively high dimensional $q$-ary functions.


The main functionality of our algorithm is availible in the `QSPRIGHT` class. Example usage is given below:

```
test_signal = get_random_signal(n=n, q=q, sparsity=sparsity, a_min=a_min, a_max=a_max, noise_sd=noise_sd)

transformer = QSPRIGHT(
    query_method="complex",
    delays_method="identity",
    reconstruct_method="noiseless",
    num_subsample=3,
    b=4
)
start_time = time.time()
ft, (n_used, n_used_unique, _), peeled = spright.transform(test_signal, verbose=False, report=True)
```

<img src="figs/complexity-vs-n-lasso-1.png" width="300">
<img src="figs/complexity-vs-n-qspright-1.png" width="300">
<img src="figs/complexity-vs-n-rna-1.png" width="300">
<img src="figs/complexity-vs-n-runtime-1.png" width="300">
<img src="figs/nmse-vs-snr-1.png" width="300">

