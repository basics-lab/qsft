# Efficient Sparse q-ary Fourier Transforms

This repository contains code for the paper:

_"Efficiently Computing Sparse Fourier Transforms of_ $q$_-ary Functions" Yigit Erginbas*, Justin Kang*, Amirali Aghazadeh, Kannan Ramchandran_

*Equal contribution: These authors contributed equally.

This package may be useful to you if you deal with complicated functions of $q$-ary sequences, for example, 
functions of protiens, DNA or RNA. 
### Table of Contents
* [Abstract](#abstract)
* [Quick Start](#quickstart)
* [Signals](#signals)
* [The QSFT Class](#qsft)
* [Test Helper](#test-helper)
* [Experimental Results](#exp)
  * [Comparing with LASSO](#LASSO)
  * [SNR vs NMSE](#snr)
  * [Example from Computational Biology](#rna)

### Abstract
<a id=abstract></a>
Fourier transformations of pseudo-Boolean functions are popular tools for analyzing functions of binary sequences. Real-world functions often have structures that manifest in a sparse Fourier transform, and previous works have shown that under the assumption of sparsity the transform can be computed efficiently. But what if we want to compute the Fourier transform of functions defined over a $q$-ary alphabet? These types of functions arise naturally in many areas including biology. A typical workaround is to encode the $q$-ary sequence in binary however, this approach is computationally inefficient and fundamentally incompatible with the existing sparse Fourier transform techniques. Herein, we develop a sparse Fourier transform algorithm specifically for $q$-ary functions of length $n$ sequences, dubbed $q$-SFT, which provably computes an $S$-sparse transform with vanishing error as $q^n$ goes to $\infty$ in $O(Sn)$ function evaluations and $O(S n^2 \log q)$ computations, where $S = q^{n\delta}$ for some $\delta < 1$. Under certain assumptions, we show that for fixed $q$, a robust version of $q$-SFT has a sample complexity of $O(Sn^2)$ and a computational complexity of $O(Sn^3)$ with the same asymptotic guarantees. We present numerical simulations on synthetic and real-world RNA data, demonstrating the scalability of $q$-SFT to massively high dimensional $q$-ary functions.

### Quick Start
<a id=quickstart></a>
The main functionality of our algorithm is available in the `QSFT` class. A minimal example can be found in 
`synt_exp/quick_example.py`. Details on how this file works can be found in other sections of the README. 

### Signals
<a id=signals></a>
In this section, we discuss the `Signal` objects that we use to interface with the `QSFT` class.
A `Signal` encapsulates the object that we are trying to transform (you may interpret it as a signal of length $q^n$ 
or a function of $n$ $q$-ary variables). Most relevant to our discussion is the 
`SubsampledSignal` class found at `qsft.input_signal_subsampled.SubsampledSignal`. This class can be extended to 
easily create a signal for the specific application that we desire. For example, we create a 
synthetic signal that is sparse in the Fourier domain in 
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
Let's break this down. 
* `subsampling_method` should be set to `qsft` if we plan to use the `QSFT` class, otherwise it 
should be set to `lasso` if LASSO will be used. 
* The `query_method` argument is set to "complex", which 
sets our subsampling matricies $M_c$ 
to be generated randomly. This works very well in practice, in particular for situations where you do not expect the 
Fourier coefficients to be uniformly distributed. Alternately, setting this argument to "simple" will generate $M_c$ 
according to the identity matrix structure in our paper, which works provably well when fourier coefficients. 
* The 
`num_subsample` parameter
sets $C$, the number of different matricies $M_c, \;c=1,\dotsc,C$ that are used. A good place to start is $C=3$, but 
you can adjust it later to potentially improve performance.
* `b` determines the inner dimension of 
the subsampling. This parameter must be chosen such that the number of non-zero coefficients is $O(q^b)$. If you 
don't know the sparsity of your signal, you may have you adjust `b` until you find a suitable value. Note that you 
don't want to make `b` too large either, as that will increase your sample and computational complexity.  

* The `delays_method_source` parameter is set to "identity". 
In general, this should be set to "identity", unless you know that the max hamming weight of the non-zero Fourier 
coefficients are low (i.e., the Fourier transform is low degree). This will use $n$ delays. If you know, however, 
that the max hamming weight (i.e., degree) of non-zero Fourier coefficients is lower than $t$ _and_ $q$ is prime, then 
you can use the "coded" setting, 
which 
uses only $2t \log_q n$ delays instead, a significant improvement when $n$ is large. This is often the 
case when the function you are dealing with represents some real-world function. 
* If you 
set `delays_method_source` to "coded", you must also include the `t` parameter. The `QSFT` class reports the max 
hamming weight of non-zero 
coefficients, so if you find that they are constantly low, consider enabling this for a significant speedup. 

With `query_args` set, we can now construct our signal object. To do so, we call the `get_random_subsampled_signal`, 
which randomly generates a `SyntheticSubsampledSingal` for us.
```python
test_signal = get_random_subsampled_signal( n=n,
                                            q=q,
                                            sparsity=sparsity,
                                            a_min=a_min,
                                            a_max=a_max,
                                            noise_sd=noise_sd,
                                            query_args=query_args,
                                            max_weight=t)
```
Some parameters are explained below:
* `n`, `q` represent the number of function inputs and alphabet size respectively (for interpretation as a signal 
  this is a signal with $q^n$ elements).
* `sparsity` is the number of non-zero coefficients that should be in the transform
* `a_min` and `a_max` are the minimum and maximum modulus of the nonzero coefficients, which is chosen uniformly 
  over this range.
* `noise_sd` is the stander deviation of the additive noise added to the signal.
* `max_weight` (optional) is the max weight of non-zero Fourier coefficients in the generated signal. The set of 
  indicies for the non-zero Fourier coefficients are chosen uniformly over all indicies with hamming weight 
  `max_weight` or less. In general, you probably want `max_weight` to be equal to `t` in query_args, since setting 
  `t` ensures you only look for coefficients with indicies of weight `t` or less.

Now that we have a signal object, the next step is to take its transform!

### QSFT
<a id=qsft></a>
Once we construct the signal we want to transform, the next step is to create the QSFT object that will perform the 
transformation. Again, we start with the key arguments for 
```python
    qsft_args = {
        "num_subsample": num_subsample,
        "num_repeat": num_repeat,
        "reconstruct_method_source": delays_method_source,
        "reconstruct_method_channel": delays_method_channel,
        "b": b,
        "noise_sd": noise_sd,
        "source_decoder": decoder
    }
```
* `num_subsample`, `num_repeat`,  and `b` are similar to the equivalent parameters for the signal object. a QSFT 
  instance may only be used on a singal if its corresponding parameters are leger or equal. For example, if we have 
  a signal with `num_subsample = 3`, we can set `num_subsample` to be any value $\leq 3$.
* `delays_method_source` and `delays_method_channel` must exactly match those of the signal you intend to use with 
  the QSFT instance. If `delays_method_source = "coded"`, you must also pass a function handle `source_decoder`. We 
  have implemented a function that returns a suitable Reed Solomon decoder in `get_reed_solomon_dec`.
* `noise_sd` is a hyperparameter that is a proxy for the amount of additive noise in the signal. If the signal is 
  truly corrupted by additive gaussian noise, using the variance of that noise is a good choice for the is parameter,
  otherwise, in a real-world setting you may have to adjust this to find a suitable value.
We can then use these values to create an instance of `QSFT`.
```python
    sft = QSFT(**qsft_args)
    result = sft.transform(test_signal, verbosity=0, timing_verbose=True, report=True, sort=True)
```
* The `verbosity` argument determines the amount of printouts, not including timing information. When it is set to 
  0 there are no printouts, when it is set to 10, the maximum number of printouts are provided.
* When `timing_verbose` is `True` information about how long each step of transform took is included.
* When `report` is set to `False` only the transform is output, when it is set to `True`, a collection of useful 
  statistics are included. The docstring of the `QSFT` class contains more information about what is included in the 
  output in this case. In `synt_exp/quick_example.py` an example is provided where the additional information is 
  processed and displayed.  
### Test Helper
<a id=test-helper></a>

The `TestHelper` is an **abstract class** used to encapsulate the complete pipeline of sampling, data storage, data loading and sparse Fourier transformation.
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

### Experimental Results
<a id=exp></a>


#### Comparing with LASSO
<a id=LASSO></a>
In addition to implementing `QSFT`, we also include a comparison with LASSO implemented via 
[group-lasso](https://github.com/yngvem/group-lasso), which is significantly slower for this application. 

The following figures compare LASSO and QSFT. These figures were generated by 
using the scripts at `synt_exp/run-tests-complexity-vs-size.py` and plotted by `synt_exp/plot-complexity-vs-size.py`.
The grey area in the first graph is a region where LASSO took too long to converge.
<p align="center">
LASSO vs. QSFT<br>
<img src="figs/complexity-vs-n-lasso-1.png" width="300">
<img src="figs/complexity-vs-n-qspright-1.png" width="300">
</p>

<p align="center">
<img src="figs/complexity-vs-n-runtime-1.png" width="300">
</p>

As we can see, the runtime of LASSO is sub-exponential in $n$ .

#### SNR vs NMSE
As the amount of noise in the signal/function increases, sucessful recover becomes more difficult. To examine this 
phenomonon, the script `synt_exp/run-tests-nmse-vs-snr.py` is useful. In graph below, we see that for different 
sparsity levels $S `QSFT` goes from a very high to low NMSE at some threshold. This type of _phase transtion_ 
behaviour is tpyical in compressed sensing.
<a id=snr></a>
<p align="center">
<img src="figs/nmse-vs-snr-1.png" width="300" alt="SNR vs NMSE">
</p>

####  Real-World Example from Computational Biology
<a id=rna></a>

This repository also provide an example of how to apply our code to a complex $q$-ary function in  
[ViennaRNA](https://github.com/ViennaRNA/ViennaRNA). 
Code for this example is in the `rna_exp` folder. We create the `RnaSubsampledSignal(SubsampledSignal)` Class. The 
`subsample(self, query_indices)` function interfaces with the ViennaRNA package, to compute the Mean Free Energy (MFE)
 of an RNA sequence.
<p align="center">
<img src="figs/complexity-vs-n-rna-1.png" width="300" alt="Example: Computational Biology">
</p>

The graph above shows that when $n$ is large, our the `QSFT` function achieves a low NMSE. This means that `QSFT` 
generates a sparse fourier transform that is able to compute the MFE of an arbitrary unseen RNA sequence with 
relatively little error. 
