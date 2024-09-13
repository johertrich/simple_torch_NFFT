# Very simple but vectorized torch implementation of the NFFT

This repository contains a very simple implementation of the non-equispaced fast Fourier transform (NFFT)
implemented directly in PyTorch. It runs on a GPU, supports autograd and vectorization (so far only wrt the basis points).

## Comments towards the State of Implementation

- so far only in 1D
- so far only batching wrt the basis points
- so far only autograd wrt f/f_hat not wrt basis points
- oversampled and non-oversampled number of Fourier coefficients should be even
- autograd not tested yet 
- more efficient with small cutoff parameters...
- pip install not set up yet

## Requirements

Just PyTorch (version >= 2.4, because otherwise torch.compile has issues with Python 3.12).
The package can be installed with

```
pip install git+https://github.com/johertrich/simple_torch_NFFT
```

## Author

[Johannes Hertrich](https://johertrich.github.io)
