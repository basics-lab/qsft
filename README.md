# Efficient Sparse q-ary Fourier Transforms

This repository contains code for the paper:

_"Efficiently Computing Sparse Fourier Transforms of_ $q$_-ary Functions" Yigit Erginbas*, Justin Kang*, Amirali Aghazadeh, Kannan Ramchandran_

### Table of Contents
* [Abstract](#abstract)
* [Quick Start](#quickstart)
* [Signals](#signals)
  * [Example: Computational Biology](#rna)
* [Comparing with LASSO](#LASSO)

### Abstract
<a id=abstract></a>
Fourier transformations of pseudo-Boolean functions are popular tools for analyzing functions of binary sequences. Real-world functions often have structures that manifest in a sparse Fourier transform, and previous works have shown that under the assumption of sparsity the transform can be computed efficiently. But what if we want to compute the Fourier transform of functions defined over a $q$-ary alphabet? These types of functions arise naturally in many areas including biology. A typical workaround is to encode the $q$-ary sequence in binary however, this approach is computationally inefficient and fundamentally incompatible with the existing sparse Fourier transform techniques. Herein, we develop a sparse Fourier transform algorithm specifically for $q$-ary functions of length $n$ sequences, dubbed $q$-SFT, which provably computes an $S$-sparse transform with vanishing error as $q^n$ goes to $\infty$ in $O(Sn)$ function evaluations and $O(S n^2 \log q)$ computations, where $S = q^{n\delta}$ for some $\delta < 1$. Under certain assumptions, we show that for fixed $q$, a robust version of $q$-SFT has a sample complexity of $O(Sn^2)$ and a computational complexity of $O(Sn^3)$ with the same asymptotic guarantees. We present numerical simulations on synthetic and real-world RNA data, demonstrating the scalability of $q$-SFT to massively high dimensional $q$-ary functions.

### Quick Start
<a id=quickstart></a>
The main functionality of our algorithm is availible in the `QSFT` class. Example usage is given below:

### Signals
<a id=signals></a>
In this section, we discuss the `Signal` objects that we use to interface with the `QSFT` class.
A `Signal` encapsulates the object that we are trying to transform (you may interpret it as a signal of length $q^n$ 
or a function of $n$ $q$-ary variables). Most relevant to our discussion is the 
`SubsampledSignal` class found at `qsft.input_signal_subsampled.SubsampledSignal`. This class can be extended to 
easily create a signal for the specific application that we desire. For example, we create a 
synthetic signal that is sparse in the fourier domain in 
`synt_exp.synt_src.synthetic_signal.SyntheticSparseSignal`. The `subsample()` function must be implemented in the 
extended class. This function takes a list of `query_indicies` and outputs a list of  fuction/signal value at the given 
query indicies. We refer to the `SyntheticSparseSignal` as an example.

We can construct a `SyntheticSparseSignal` as follows. First, we need to declare the `query_args`:

```python
    query_args = {
        "subsampling_method": "qsft",
        "query_method": "complex",
        "num_subsample": num_subsample,
        "b": b,
        "delays_method_source": "identity",
        "delays_method_channel": "nso",
        "num_repeat": num_repeat,
        "t": t
    }
```
Let's break this down. `subsampling_method` should be set to `qsft` if we plan to use the `QSFT` class, otherwise it 
should be set to `lasso` if LASSO will be used. The `query_method` argument is set to "complex", which 
sets our subsampling matricies $M_c$ 
to be generated randomly. This works very well in practice, in particular for situations where you do not expect the 
Fourier coefficients to be uniformly distributed. Alternately, setting this argument to "simple" will generate $M_c$ 
according to the identity matrix structure in our paper, which works provably well when fourier coefficients. The 
`num_subsample` parameter
sets $C$, the number of different matricies $M_c, c=1,\dotsc,C$ that are used. `b` determines the inner dimension of 
the subsampling. This parameter must be chosen such that the number of non-zero coefficients is $O(q^b)$.

Next are the parameters related to the delay structure. The `delays_method_source` parameter is set to "identity". 
In general, this should be set to "identity", unless you know that the max hamming weight of the non-zero fourier 
coefficients are low (i.e., the Fourier transform is low degree). This will use $n$ delays. If you know, however, 
that the max hamming weight (i.e., degree) is lower _and_ $q$ is prime, then you can use the "coded" setting, which 
uses only $2t\left 
\lceil \log_q n \right \rceil$ delays instead, a potential significant improvement when $n$ is large.



<p align="center">
<img src="figs/nmse-vs-snr-1.png" width="300">
</p>

####  Example: Computational Biology
<a id=rna></a>

<p align="center">
<img src="figs/complexity-vs-n-rna-1.png" width="300">
</p>

### Comparing with LASSO
<a id=LASSO></a>

<p align="center">
<img src="figs/complexity-vs-n-lasso-1.png" width="300">
</p>



<p align="center">
<img src="figs/complexity-vs-n-qspright-1.png" width="300">
</p>
<p align="center">
<img src="figs/complexity-vs-n-runtime-1.png" width="300">
</p>
