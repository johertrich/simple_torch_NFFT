# NFFT Implementation

Back to: [Main Page](../../readme.md)

We give a short overview on the NFFT implementation. It runs on a GPU, supports autograd (wrt both, function values and basis points) and allows batching. 

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

- A (very simple) runtime comparison with other libraries is available [here](runtime_comparison.md).
- Compared to other libraries the run time might be a little bit more sensitive towards the window size `m`.
- In contrast to other NFFT libraries there are almost no precomputations. Only the Fourier coefficients of the window functions are computed during initialization of the NFFT object. This might be a disadvantage if one often computes the NFFT with the same basis points.
- parallelization of the adjoint wrt `f` and `f_hat` is poor at the moment. I will investigate why.

### Issues that I am Aware of

- `torch.compile` throws a warning regarding complex tensors. It seems to be fine to ignore it.

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
