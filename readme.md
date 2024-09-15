# Very simple but vectorized torch implementation of the NFFT

This repository contains a very simple implementation of the non-equispaced fast Fourier transform (NFFT)
implemented directly in PyTorch for arbitrary dimensions. It runs on a GPU, supports autograd and batching.
In contrast to other NFFT libraries there are almost no precomputations. Only the Fourier coefficients of the window functions
are computed during initialization of the NFFT object.

## Comments towards the State of Implementation

- so far only autograd wrt f/f_hat not wrt basis points
- autograd not working yet
- more efficient with small cutoff parameters...
- so far no option to opt out from compiling (which might be convenient when calling many NFFT with different input sizes).

## Requirements

Just PyTorch (version >= 2.4, because otherwise torch.compile has issues with Python 3.12).
The package can be installed with

```
pip install git+https://github.com/johertrich/simple_torch_NFFT
```

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

To use the NFFT, we first have to create an NFFT object, which takes as an input the size `N` of the equidistant
grid resulting in the constructor `nfft = NFFT(N)`. Optional arguments are given in the example below.

The NFFT object provides the forward and adjoint NFFT approximating the forward and adjoint problem from above.
Here, the forward NFFT takes as argument the points `x` and the function values `f_hat` and returns the result `f` and the
adjoint NFFT takes the inputs `x` and `f` and returns `f_hat`. Thus, the resulting function calls are `nfft(x,f)` 
(or equivalently `nfft.forward(x,f_hat)`) and `nfft.adjoint(x,f)`.

All of these tensors have as the first dimension the batch dimension wrt `x`, as a second dimension the batch dimension wrt `f`.
Consequently, `x` has size `(batch_x,1,M)`, `f_hat` has size `(batch_x,batch_f,N)` (as input broadcastable with size `(1,batch_f,N)`) and `f` has size `(batch_x,batch_f,M)` 
(as input broadcastable with size `(1,batch_f,M)`). The entries of `f_hat` always start with the negative index `-N/2`, so you want to start with
zero you have to use `torch.fft.ifftshift`.

Forward and adjoint NFFT will be compiled at the first call.


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
# basis points, NFFT will be taken wrt the second dimension
x = (torch.rand((batch_x, 1, M, len(N),), device=device,) - 0.5 )

# forward NFFT
f_hat_shape = [batch_x, batch_f] + list(N)  # f_hat has batch dimensions + grid dimensions
f_hat = torch.randn(f_hat_shape, dtype=torch.complex64, device=device)  # Fourier coefficients
f = nfft(x, f_hat)

# adjoint NFFT
f = torch.randn((batch_x, batch_f, M), dtype=torch.complex64, device=device)  # function values
f_hat = nfft.adjoint(x, f)

```

## Author

[Johannes Hertrich](https://johertrich.github.io)
