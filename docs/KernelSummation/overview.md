# Fast Kernel Summation via Slicing

Back to: [Main Page](../../readme.md)

In the following, we give details on the implemented fast kernel summation. A theoretical background can be found [here](background.md). A precise specification of all attributes and arguments in this library is given [here](specification.md).
We aim to compute the kernel sums

$$s_m=\sum_{n=1}^N w_n K(x_n,y_m)$$

for all $m=1,...,M$. The naive implementation has a computational complexity of $O(MN)$. The implementation of this library has complexity $O(M+N)$.

## Supported Kernels

The implementation currently supports the following kernels:

- The `Gauss` kernel given by $K(x,y)=\exp(-\frac1{2\sigma^2}\\|x-y\\|^2)$
- The `Laplace` kernel given by $K(x,y)=\exp(-\frac1{\sigma}\\|x-y\\|)$. In the literature, the parameter $\sigma$ is often replaced by $\frac{1}{\alpha}$. However, we stick to the above notation to handle the scale parameter similarly for all kernels.
- The `Matern` kernel given by $K(x,y)=\frac{2^{1-\nu}}{\Gamma(\nu)}(\tfrac{\sqrt{2\nu}}{\sigma}\\|x-y\\|)^\nu K_\nu(\tfrac{\sqrt{2\nu}}{\sigma}\\|x-y\\|)$, where $K_\nu$ is the modified Bessel function of second kind. The kernel depends on a smoothness parameter $\nu$ which determines its smoothness. For $\nu=\frac12$, we obtain the Laplace kernel and for $\nu\to\infty$, the Matern kernel converges towards the Gauss kernel.
- The `energy` kernel given by $K(x,y)=-\frac{1}{\sigma}\\|x-y\\|$. So far the custom backward pass is not implemented (i.e. it only works for `batched_autograd==False`).`


## Usage and Example

To use the fast kernel summation, we first create a `Fastsum` object with `fastsum=Fastsum(d, kernel="Gauss")`. It takes the dimension and the kernel (as string from the above list) as input. Other optional arguments are given in the example below and [here](specification.md).

Afterwards, we can compute the vector $s=(s_1,...,s_M)$ by `s=fastsum(x, y, w, xis_or_P)` where `x` has the shape `(N,d)`, `y` has the shape `(M,d)` and `w` has the shape `(N,)`. The argument `xis_or_P` either takes the number of considered slices as integer (higher number = higher accuracy) or the slices itself as a tensor of size `(P,d)`.

```python
import torch
from simple_torch_NFFT import Fastsum

device = "cuda" if torch.cuda.is_available() else "cpu"

d = 10 # data dimension
kernel = "Gauss" # kernel type


fastsum = Fastsum(d, kernel=kernel, device=device) # fastsum object
# other optional keyword arguments (with sensible defaults) are:
# kernel_params: dict with further paramters 
#   (e.g. dict(nu=1.5) for the Matern kernel with smoothness nu=1.5)
# n_ft: Number of coefficients to truncate the Fourier series of the kernel
# nfft: nfft object (for setting custom parameters in the NFFT)
# x_range: cutoff parameter for rescaling
# batch_size_P: batch size for the slices
# batch_size_nfft: batch size for the NFFT
# device: specify device
# no_compile: default False, set True to skip compile of the NFFT 
#   (useful for debugging)
# batched_autodiff: default True, set False to use backprop through 
#   the forward pass instead of overwriting the backward pass 
#   (not efficient, but sometimes useful for debugging)


scale = 1.0 # kernel parameter

P = 256 # number of projections for slicing
N, M = 10000, 10000 # Number of data points

# data generation
x = torch.randn((N, d), device=device, dtype=torch.float)
y = torch.randn((M, d), device=device, dtype=torch.float)
x_weights = torch.rand(x.shape[0]).to(x)

kernel_sum = fastsum(x, y, x_weights, scale, P) # compute kernel sum
```

## Perspectives

In the future, I want to add:

- other slicing rules
- other kernels (energy, thin plate spline, logarithmic)
- 1D Summation via KeOps as alternative to NFFT (could be useful in particular, when we are considering a low number of (relevant) Fourier features like, e.g., in the Gaussian kernel)
- for $d=2$ and $d=3$: add fast Fourier summation (without slicing)
