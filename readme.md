# PyTorch Implementation of the NFFT and Fast Kernel Sums

In this library, we implement the following methods:

- **Non-equispaced fast Fourier transform (NFFT)**: We implement the NFFT directly in PyTorch for arbitrary dimensions. It runs on a GPU, supports autograd (wrt both, function values and basis points)
and allows batching. 

- **Fast Kernel Summations via Slicing**: We apply the NFFT for the computation of large kernel sums in arbitrary dimensions.

Link to the github repository: [https://github.com/johertrich/simple_torch_NFFT](https://github.com/johertrich/simple_torch_NFFT)

## Contents

For the NFFT:

- [Overview](docs/NFFT/overview.md) of the NFFT implementation
- [Simple runtime comparison](docs/NFFT/runtime.md) of different NFFT libraries
- [Specification](docs/NFFT/specification.md) of the implemented classes and functions

For the fast kernel summation:

- [Overview](docs/KernelSummation/overview.md) of the implementation for the fast kernel summation
- [Backgrounds](docs/KernelSummation/background.md) of fast kernel summation via slicing and NFFTs (including the efficient evaluation of derivatives)
- [Specification](docs/KernelSummation/specification.md) of the implemented classes and functions 

## Installation

The library requires PyTorch (version >= 2.5 recommended) and Numpy. Then it can be installed by:
```
pip install git+https://github.com/johertrich/simple_torch_NFFT
```

## Examples

### NFFT Example

```python
import torch
from simple_torch_NFFT import NFFT

device = "cuda" if torch.cuda.is_available() else "cpu"

N = (2**10,)  # size of the regular grid as tuple, here (in 1D) 1024.

# create NFFT object
nfft = NFFT(N, device=device)

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

### Fast Kernel Summation

```python
import torch
from simple_torch_NFFT import Fastsum

device = "cuda" if torch.cuda.is_available() else "cpu"

d = 10 # data dimension
kernel = "Gauss" # kernel type
fastsum = Fastsum(d, kernel=kernel, device=device) # fastsum object
scale = 1.0 # kernel parameter

P = 256 # number of projections for slicing
N, M = 10000, 10000 # Number of data points

# data generation
x = torch.randn((N, d), device=device, dtype=torch.float)
y = torch.randn((M, d), device=device, dtype=torch.float)
x_weights = torch.rand(x.shape[0]).to(x)

kernel_sum = fastsum(x, y, x_weights, scale, P) # compute kernel sum
```

## Citation

This library was written by [Johannes Hertrich](https://johertrich.github.io) in the context of fast kernel summations via slicing.
If you find it usefull, please consider to cite

```
@article{HJQ2024,
  title={Fast Summation of Radial Kernels via {QMC} Slicing},
  author={Hertrich, Johannes and Jahn, Tim and Quellmalz, Michael},
  journal={arXiv preprint arXiv:2410.01316},
  year={2024}
}
```

or

```
@article{H2024,
  title={Fast Kernel Summation in High Dimensions via Slicing and {F}ourier transforms},
  author={Hertrich, Johannes},
  journal={SIAM Journal on Mathematics of Data Science},
  volume={6},
  number={4},
  pages={1109--1137},
  year={2024}
}
```
