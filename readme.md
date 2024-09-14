# Very simple but vectorized torch implementation of the NFFT

This repository contains a very simple implementation of the non-equispaced fast Fourier transform (NFFT)
implemented directly in PyTorch. It runs on a GPU, supports autograd and vectorization.
In contrast to other NFFT libraries there are almost no precomputations. Only the Fourier coefficients of the window functions
are computed during initialization of the NFFT object.

## Comments towards the State of Implementation

- so far only in 1D
- so far only autograd wrt f/f_hat not wrt basis points
- autograd not tested yet (probably it contains some typos)
- more efficient with small cutoff parameters...

## Requirements

Just PyTorch (version >= 2.4, because otherwise torch.compile has issues with Python 3.12).
The package can be installed with

```
pip install git+https://github.com/johertrich/simple_torch_NFFT
```

## Usage

There exist different conventions to define the Fourier transform and normalizing (N)FFTs.
In order to specify the functionality of the code precisely, we first have to define
the problem setting. Afterwards, we have a look on the precise implementation

### Problem Setting: Fourier Transform on Non-equispaced Grids

To fix the conventions and normalizations of the package, we briefly recall the definition of the
Fourier transform on non-equispaced grids. For convenience, we stick to the 1D case.
The forward problem is given by computing
$$
f_j=f(x_j)=\sum_{k=-N/2}^{N/2-1} \hat f_k e^{-2\pi k x_j},\quad j=1,...,M,
$$
where $x_1,...,x_M\in[-\frac{1}{2},\frac{1}{2})$.
The mapping $\hat f \mapsto f$ is linear and admits the adjoint operator
$$
\hat f_k=\sum_{j=1}^{M} f_j e^{2\pi k x_j},\quad k=-N/2,...,N/2-1.
$$
The NFFT, as implemented in this package, approximates these two problems by using an interpolation with
a windowing function.

### Implementation

The NFFT can be called as follows.

```python
import torch
from simple_torch_NFFT import NFFT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters of the NFFT
N = 2**10  # number of Fourier coefficients

# create NFFT object
nfft = NFFT(N)
# optional arguments are
# m (window size, default 4)
# n (oversampled number of Fourier coefficients, None for using oversampling factor, default: None)
# sigma (oversampling factor, default 2, not used if n is given)
# window (default simple_torch_NFFT.KaiserBesselWindow, other option simple_torch_NFFT.GaussWindow)
# device (default "cuda" if torch.cuda.is_available() else "cpu")
# double_precision (default false)

# Parameters of the input
M = 20000  # number of basis points
batch_x = 2  # batches of basis points
batch_f = 2  # batches of function values
# basis points, NFFT will be taken wrt the second dimension
k = (torch.rand((batch_x, 1, M,), device=device,) - 0.5 )

# forward NFFT
f_hat = torch.randn(
    (k.shape[0], batch_f, N), dtype=torch.complex64, device=device
)  # Fourier coefficients
f = nfft(k, f_hat)

# adjoint NFFT
f = torch.randn(k.shape, dtype=torch.complex64, device=device)  # function values
f_hat = nfft.adjoint(k, f)

```

## Author

[Johannes Hertrich](https://johertrich.github.io)
