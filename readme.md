# Simple PyTorch Implementation of the NFFT

This repository contains a very simple implementation of the non-equispaced fast Fourier transform (NFFT)
implemented directly in PyTorch for arbitrary dimensions. It runs on a GPU, supports autograd (wrt both, function values and basis points)
and allows batching. 

## Requirements

Just PyTorch and NumPy are required.
The package can be installed with

```
pip install git+https://github.com/johertrich/simple_torch_NFFT
```

By default, the package uses `torch.compile` which is not well supported for older versions of PyTorch and might
cause issues with older GPUs.
PyTorch version 2.4 (and newer) are recommended. 
In order to run the compiled version on a GPU, CUDA compatibility, >= 7 will be required.
The `torch.compile` can be deactivated in the constructor of the NFFT object (with `no_compile=True`).

## Usage

There exist different conventions to define the Fourier transform and for normalizing (N)FFTs.
In order to specify the functionality of the code precisely, we first have to define
the problem setting. Afterwards, we have a look on the precise implementation and give an example.

### Problem Setting: Fourier Transform on Non-equispaced Grids

To fix the conventions and normalizations of the package, we briefly recall the definition of the
Fourier transform on non-equispaced grids. Here, we only consider the cases "equidistant to non-equidistant" and
"non-equidistant to equidistant" but not "non-equidistant to non-equidistant". For convenience, we stick to the 1D case.
The forward problem is given by computing

$$f_j=f(x_j)=\sum_{k=-N/2}^{N/2-1} \hat f_k e^{-2\pi k x_j},\quad j=1,...,M,$$

where $x_1,...,x_M\in[-\frac{1}{2},\frac{1}{2})$.
The mapping $\hat f \mapsto f$ is linear and admits the adjoint operator

$$\hat f_k=\sum_{j=1}^{M} f_j e^{2\pi k x_j},\quad k=-N/2,...,N/2-1.$$

The NFFT, as implemented in this package, approximates these two problems by using an interpolation with
a windowing function.

### Implementation

To use the NFFT, we first have to create an NFFT object, which takes as an input the size `N=(N_1,...,N_d)` of the equidistant
grid resulting in the constructor `nfft = NFFT(N)`. Optional arguments are given in the example below.

The NFFT object provides the forward and adjoint NFFT approximating the forward and adjoint problem from above.
Here, the forward NFFT takes as argument the points `x` and the function values `f_hat` and returns the result `f` and the
adjoint NFFT takes the inputs `x` and `f` and returns `f_hat`. Thus, the resulting function calls are `nfft(x,f)` 
(or equivalently `nfft.forward(x,f_hat)`) and `nfft.adjoint(x,f)`.

For testing reasons there exists also a class `NDFT` which explicitly computes the forward and backward problem. A NDFT object can
be created by `ndft = NDFT(N)` and implements the same `ndft.forward(x,f_hat)` and `ndft.adjoint(x,f)` methods.

All of these tensors support arbitrary batch dimensions. That is, `x` has size `(...,M,d)`, `f_hat` has size `(...,N)` 
(or `(...,N_1,N_2)` for 2D, `(...,N_1,N_2,N_3)` for 3D and so on) and `f` has size `(...,M)`, 
where `...` refers to the batch dimensions. For calling `forward(x,f_hat)` (or `adjoint(x,f)` respectively), the batch dimensions of the two inputs
have to be equal or broadcastable. The entries of `f_hat` always start with the negative index `-N/2`, so you want to start with
zero you have to use `torch.fft.ifftshift`.

Forward and adjoint NFFT will be compiled at the first call, despite you pass `no_compile=True` to the constructor.


### Example

```python
import torch
from simple_torch_NFFT import NFFT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters of the NFFT
N = (2**10,)  # size of the regular grid as tuple, here (in 1D) 1024.

# create NFFT object
nfft = NFFT(N)
# optional arguments are
# m (window size, default 4)
# n (size of the oversampled grid (tuple), None for using oversampling factor, default: None)
# sigma (oversampling factor, default 2, not used if n is given)
# window (default simple_torch_NFFT.KaiserBesselWindow, other option simple_torch_NFFT.GaussWindow)
# device (default "cuda" if torch.cuda.is_available() else "cpu")
# double_precision (default false)
# no_compile (set to True to supress compile, mainly useful for debugging to get readible stack traces, default: False)

# Parameters of the input
M = 20000  # number of basis points
batch_x = 2  # batches of basis points
batch_f = 2  # batches of function values
# basis points, NFFT will be taken wrt the last dimension
x = (torch.rand((batch_x, 1, M, len(N),), device=device,) - 0.5 )

# forward NFFT
f_hat_shape = [batch_x, batch_f] + list(N)  # f_hat has batch dimensions + grid dimensions
f_hat = torch.randn(f_hat_shape, dtype=torch.complex64, device=device)  # Fourier coefficients
f = nfft(x, f_hat)

# adjoint NFFT
f = torch.randn((batch_x, batch_f, M), dtype=torch.complex64, device=device)  # function values
f_hat = nfft.adjoint(x, f)

```

## Comments

I wrote this implementation as some kind of exercise for myself, but actually it works very well.

### Performance Comments

- Compared to other libraries the run time might be a little bit more sensitive towards the window size `m`.
- In contrast to other NFFT libraries there are almost no precomputations. Only the Fourier coefficients of the window functions are computed during initialization of the NFFT object. This might be a disadvantage if one often computes the NFFT with the same basis points.
- parallelization of the adjoint wrt `f` and `f_hat` is poor at the moment. I will investigate why.

### Runtime Comparison

The test script `test/tuntime_test.py` compares the runtime of this implementation with some other libraries.
For the comparisons, we always perform all precomputations, which only depend on the input shape and dimension,
but no precomputations depending on the basis points `x` or the function values `f` or `f_hat`.
We fix `M=100000`, `m=2` and `sigma=2`. Then, we obtain the following execution times on a NVIDIA RTX 4090 GPU averaged over 10 runs. 
Moreover, we run all methods with single precision. Note that pyNFFT3 only has a CPU implementation (24-core AMD Ryzen Threadripper 7960X s) and
always uses double precision such that the comparison with it is not really fair.

We always use the input shape `(batch_x,1,M,d)` for `x`, `(batch_x,batch_f,M)` for `f` and `(batch_x,batch_f,N_1,...,N_d)` for `f_hat`.

#### One-dimensional NFFT

We use `N=4096`. Then the execution times (seconds) for 10 the forward NFFT were the following.

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.00226 | 0.00113 | 0.00140 | 0.00078 |
| `batch_x=1`, `batch_f=10` | 0.00505 | 0.00108 | 0.00333 | 0.00096 |
| `batch_x=10`, `batch_f=1` | 0.01340 | 0.00447 | 0.00346 | 0.00085 |
| `batch_x=10`, `batch_f=10` | 0.06744 | 0.00500 | 0.02241 | 0.00288 |

For the adjoint NFFT, we obtain the following execution times

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.00289 | 0.00270 | 0.00156 | 0.00091 |
| `batch_x=1`, `batch_f=10` | 0.01650 | 0.00289 | 0.00325 | 0.00183 |
| `batch_x=10`, `batch_f=1` | 0.02491 | 0.00613 | 0.00310 | 0.00183 |
| `batch_x=10`, `batch_f=10` | 0.15513 | 0.00804 | 0.02222 | 0.01483 |

#### Two-dimensional NFFT

We use `N=(N_1,N_2)=(256,256)`. Then the execution times (seconds) for the forward NFFT were the following.

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.00489 | 0.00237 | 0.00257 | 0.00092 |
| `batch_x=1`, `batch_f=10` | 0.02079 | 0.00263 | 0.00973 | 0.00168 |
| `batch_x=10`, `batch_f=1` | 0.07189 | 0.01557 | 0.01020 | 0.00240 |
| `batch_x=10`, `batch_f=10` | 0.21706 | 0.02359 | 0.06724 | 0.01905 |

For the adjoint NFFT, we obtain the following execution times

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.00491 | 0.00502 | 0.00223 | 0.00117 |
| `batch_x=1`, `batch_f=10` | 0.02924 | 0.00649 | 0.00893 | 0.00725 |
| `batch_x=10`, `batch_f=1` | 0.07695 | 0.02437 | 0.00950 | 0.00791 |
| `batch_x=10`, `batch_f=10` | 0.30029 | 0.03246 | 0.06537 | 0.07389 |

#### Three-dimensional NFFT

We use `N=(N_1,N_2,N_3)=(128,128,128)`. For `batch_x=batch_f=10`, we got an memory error on the GPU 
(should not be surprising when trying to perform 100 three-dimensional NFFTs in parallel).
Then the execution times (seconds) for the forward NFFT were the following.

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.03677 | 0.00888 | 0.01297 | 0.00458 |
| `batch_x=1`, `batch_f=10` | 0.27084 | 0.02778 | 0.06780 | 0.04576 |
| `batch_x=10`, `batch_f=1` | 0.30655 | 0.08501 | 0.06860 | 0.04744 |

For the adjoint NFFT, we obtain the following execution times

| batch sizes | pyNFFT3 (CPU) | TorchKbNufft | torch_nfft | simple_torch_NFFT |
| :---: | :---: | :---: | :---: | :---: |
| `batch_x=1`, `batch_f=1` | 0.04423 | 0.01856 | 0.00950 | 0.00675 |
| `batch_x=1`, `batch_f=10` | 0.34766 | 0.04173 | 0.10051 | 0.06409 |
| `batch_x=10`, `batch_f=1` | 0.38263 | 0.11934 | 0.10062 | 0.06685 |

### Issues that I am Aware of

- `torch.compile` throws a couple of warnings on the GPU. But it works...

## Other Libraries

I am aware that there are various other NFFT libraries. You can find a list below (probably I missed some, feel free to point me to them).

### CPU and GPU

- [TorchKbNufft](https://github.com/mmuckley/torchkbnufft): high-level library written in PyTorch

### CPU only

- [pyNFFT3](https://github.com/NFFT/pyNFFT3): official wrapper for [NFFT3](https://www-user.tu-chemnitz.de/~potts/nfft/)
- [pyNFFT](https://github.com/pyNFFT/pyNFFT): other (inofficial?) wrapper for [NFFT3](https://www-user.tu-chemnitz.de/~potts/nfft/)
- [FINUFFT](https://github.com/flatironinstitute/finufft): written in C++ with python-wrapper, GPU-variant below

### GPU only

- [torch_nfft](https://github.com/dominikbuenger/torch_nfft): written in CUDA with python-wrapper
- [cuFINUFFT](https://github.com/flatironinstitute/cufinufft/): written in CUDA/C++ with python-wrapper, CPU variant above

### Tools and other Languages

- [Bindings-NUFFT](https://github.com/albangossard/Bindings-NUFFT-pytorch): Collection of PyTorch bindings and autograd definitions for different NFFT libraries
- [NFFT3](https://www-user.tu-chemnitz.de/~potts/nfft/): High performance library written in C, provides interfaces for [Python](https://github.com/NFFT/pyNFFT3) (see above), [Julia](https://github.com/NFFT/NFFT3.jl) and [Matlab](https://www-user.tu-chemnitz.de/~potts/nfft/download.php).
- [NFFT.jl](https://github.com/JuliaMath/NFFT.jl): Julia Library with CUDA support


## Author

This library was written by [Johannes Hertrich](https://johertrich.github.io) in the context of fast kernel summations via slicing.
If you find it usefull, please consider to cite

```
@article{HJQ2024,
  title={Fast Summation of Radial Kernels via {QMC} Slicing},
  author={Hertrich, Johannes and Jahn, Tim, and Quellmalz Michael},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

or

```
@article{H2024,
  title={Fast kernel summation in high dimensions via slicing and {F}ourier transforms},
  author={Hertrich, Johannes},
  journal={SIAM Journal on Mathematics of Data Science, to appear},
  year={2024}
}
```
