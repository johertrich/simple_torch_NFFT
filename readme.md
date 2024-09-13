# Very simple but vectorized torch implementation of the NFFT

This repository contains a very simple implementation of the non-equispaced fast Fourier transform (NFFT)
implemented directly in PyTorch. It runs on a GPU, supports autograd and vectorization (so far only wrt the basis points).
In contrast to other NFFT libraries there are almost no precomputations. Only the Fourier coefficients of the window functions
are computed during initialization of the initialization of the NFFT object.

## Comments towards the State of Implementation

- so far only in 1D
- so far only batching wrt the basis points
- so far only autograd wrt f/f_hat not wrt basis points
- oversampled and non-oversampled number of Fourier coefficients should be even
- autograd not tested yet 
- more efficient with small cutoff parameters...

## Requirements

Just PyTorch (version >= 2.4, because otherwise torch.compile has issues with Python 3.12).
The package can be installed with

```
pip install git+https://github.com/johertrich/simple_torch_NFFT
```

## Usage

```python
from simple_torch_NFFT import NFFT

# Parameters of the NFFT
N = 2**10 # number of Fourier coefficients
m = 4 # window size
sigma = 2 # oversampling ratio

# create NFFT object
nfft = NFFT(N, m, sigma) 
# optional arguments are 
# window (default simple_torch_NFFT.KaiserBesselWindow, other option simple_torch_NFFT.GaussWindow)
# device (default "cuda" if torch.cuda.is_available() else "cpu")
# double_precision (default false)

# Parameters of the input
M = 20000 # number of basis points
batch_dimension = 2 # batches of basis points
# basis points, NFFT will be taken wrt the second dimension
k = (torch.rand((batch_dimension,J,),device=device,dtype=float_type,) - 0.5) 

# forward NFFT
f_hat = torch.randn((k.shape[0], N), dtype=complex_type, device=device)
f = nfft(k, f_hat)

# adjoint NFFT
f = torch.randn(k.shape, dtype=complex_type, device=device) # function values
f_hat = nfft.adjoint(k, f)

```

## Author

[Johannes Hertrich](https://johertrich.github.io)
