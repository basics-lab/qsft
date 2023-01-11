# Efficient Sparse q-ary Fourier Transforms

This repository contains code for the paper:

_"Efficiently Computing Sparse Fourier Transforms of_ $q$_-ary Functions" Yigit Erginbas*, Justin Kang*, Amirali Aghazadeh, Kannan Ramchandran_

*Equal contribution: These authors contributed equally.

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
that the max hamming weight (i.e., degree) is lower than $t$ _and_ $q$ is prime, then you can use the "coded" setting, 
which 
uses only $2t \log_q n $ delays instead, a potential significant improvement when $n$ is large. Note that if you 
choose "coded", you must also include the `t` parameter.

Finally, 

<p align="center">
<img src="figs/nmse-vs-snr-1.png" width="300" alt="SNR vs NMSE">
</p>

####  Example: Computational Biology
<a id=rna></a>

<p align="center">
<img src="figs/complexity-vs-n-rna-1.png" width="300" alt="Example: Computational Biology">
</p>

### Test Helper

The `TestHelper` is an **abstract class** used to encapsulate the complete pipeline of sampling, data storage, data loading and spare Fourier transformation.
It contains a single abstract method `generate_signal` that needs to be overriden when inheriting `TestHelper`.

The only argument of the `generate_signal` method is the dictionary `signal_args` that is provided to the helper object at object creation.
The `generate_signal` method needs to be implemented such that for a given `signal_args` dictionary, it returns the corresponding `Signal` object.

For instance, the `SynthethicHelper` class inherits `TestHelper` and overrides the `generate_signal` method as follows.
```python
from qsft.test_helper import TestHelper
from synt_exp.synt_src.synthetic_signal import SyntheticSubsampledSignal

class SyntheticHelper(TestHelper):
    def generate_signal(self, signal_args):
        return SyntheticSubsampledSignal(**signal_args)
```

Then a `SyntheticHelper` object needs be created with following arguments:

```python
TestHelper(signal_args,
           methods, 
           subsampling_args,
           test_args,
           exp_dir)
```

Here, the arguments are as follows:
* `signal_args` argument is directly provided to `generate_signal` method and used to generate `Signal` objects. 
* The `methods` argument is a list of Strings that determines which algorithms are going to be used with the helper object.
Possible options are `"qsft"`, `"coded_qsft"` and `"lasso"`. 
* The `subsampling_args` argument is a dictionary that contains `num_subsample` (number of different subsampling matrices), `num_repeat` (number of repetitions in coding), `b` (inner dimension of subsampling).
* The `test_args` argument is a dictionary that contains `n_samples` (number of test samples).
* The `exp_dir` argument is an optional argument that specifies the directory to save the samples and sub-transforms for later usage. If provided directory contains previously computed samples and sub-transforms, they are loaded instead of computing again.

For instance, the following code creates a `SyntheticHelper` object
```python
methods = ["qsft"]
subsampling_args = {
            "num_subsample": 5,
            "num_repeat": 3,
            "b": 7,
        }
test_args = { "n_samples": 200000 }
helper = SyntheticHelper(signal_args, methods, subsampling_args, test_args, exp_dir)
```

At the time of object creation, the signal object is generated and subsampled. To compute the model using samples, we call `compute_model` method with arguments
* `method`: algorithm to be used. Possible options are `"qsft"`, `"coded_qsft"` and `"lasso"`.
* `model_kwargs`: If `method` is `"qsft"` or `"coded_qsft"`, it needs to be a dictionary with fields `"num_subsample"`, `"num_repeat"`, `"b"`, and `"noise_sd"` (standard deviation of the noise, it is used to determine the threshold for bin identification). The values for `"num_subsample"`, `"num_repeat"`, `"b"` must be less than or equal to the values in `signal_args` provided to the `TestHelper` object at the time of creation. Even if sampling is done for larger values, we can compute the models for lower values of these arguments using a subset of the samples. 
If `method` is `"lasso"`, it needs to be a dictionary with fields `"n_samples"` (the number of uniformly chosen samples) and `"noise_sd"`.

For instance, we can run
```python
method = "qsft"
model_kwargs = {
            "num_subsample": 2,
            "num_repeat": 2,
            "b": 7,
            "noise_sd": 0.01
}
helper.compute_model(method, model_kwargs)
```

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
