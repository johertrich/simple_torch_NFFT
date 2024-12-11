# Fast Kernel Summation via Slicing

Back to: [Main Page](../../readme.md)

In the following, we give details on the implemented fast kernel summation. A theoretical background can be found [here](background.md). A precise specification of all attributes and arguments in this library is given [here](specification.md).
We aim to compute the kernel sums

$$s_m=\sum_{n=1}^N w_n K(x_n,y_m)$$

for all $m=1,...,M$. The naive implementation has a computational complexity of $O(MN)$. The implementation of this library has complexity $O(M+N)$.

## Supported Kernels

The implementation currently supports the following kernels:

- The `Gauss` kernel given by $K(x,y)=\exp(-\frac1{2\sigma^2}\\|x-y\\|^2)$.
- The `Laplace` kernel given by $K(x,y)=\exp(-\frac1{\sigma}\\|x-y\\|)$. In the literature, the parameter $\sigma$ is often replaced by $\frac{1}{\alpha}$. However, we stick to the above notation to handle the scale parameter similarly for all kernels.
- The `Matern` kernel given by $K(x,y)=\frac{2^{1-\nu}}{\Gamma(\nu)}(\tfrac{\sqrt{2\nu}}{\sigma}\\|x-y\\|)^\nu K_\nu(\tfrac{\sqrt{2\nu}}{\sigma}\\|x-y\\|)$, where $K_\nu$ is the modified Bessel function of second kind. The kernel depends on a smoothness parameter $\nu$ which determines its smoothness. For $\nu=\frac12$, we obtain the Laplace kernel and for $\nu\to\infty$, the Matern kernel converges towards the Gauss kernel.
- The `energy` kernel given by $K(x,y)=-\frac{1}{\sigma}\\|x-y\\|$.
- The `thin_plate` spline kernel is given by $K(x,y)=\frac{\\|x-y\\|^2}{\sigma^2}\log(\frac{\\|x-y\\|}{\sigma})$.
- The `logarithmic` kernel is given by $K(x,y)=\log(\frac{\|x-y\|}{\sigma})$. For this kernel holds that $K(x,y)\to-\infty$ for $x-y\to0$. Therefore, the fast Fourier summation requires significantly more Fourier coefficients then for the other kernels (e.g., `n_ft=65536` for a relative error up to `3e-3`). A better treatment of this issue is implemented in the [NFFT3 library](https://www-user.tu-chemnitz.de/~potts/nfft/).


## Usage and Example

To use the fast kernel summation, we first create a `Fastsum` object with `fastsum=Fastsum(d, kernel="Gauss")`. It takes the dimension and the kernel (as string from the above list) as input. 

Afterwards, we can compute the vector $s=(s_1,...,s_M)$ by `s=fastsum(x, y, w, xis_or_P)` where `x` has the shape `(N,d)`, `y` has the shape `(M,d)` and `w` has the shape `(N,)`. The argument `xis_or_P` either takes the number of considered slices as integer (higher number = higher accuracy) or the slices itself as a tensor of size `(P,d)`.

Other optional arguments for the constructor of the `Fastsum` object include (full list in the [specification](specification.md)):

- `kernel_params`: For the Matern kernel, we also have to specify `kernel_params=dict(nu=nu_val)`, where `nu_val` is a float specifying the smoothness parameter $\nu$.
- Batch sizes: If the memory consumption is too high, the computation can be batched with two batch size parameters:
	- `batch_size_P`: batch size for the slices
	- `batch_size_nfft`: batch size for the NFFT (should be smaller or equal `batch_size_P`)
- `slicing_mode`: By default the slices in the [slicing algorithm](background.md) are chosen iid. The performance can often be increased by using QMC rules. Currently the modes `"iid"`, `"orthogonal"`, `"Sobol"`, `"distance"` and `"spherical_design"` are implemented, see [specification](specification.md) for a description. This choice `"spherical_design"` is only applicable for $d=3$ and $d=4$. The default value is `"spherical_design"` for $d\in\\{3,4\\}$, `"distance"` for $d \leq 100$ but $d\not\in\\{3,4\\}$ and `"orthogonal"` otherwise.



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

## Perspectives

In the future, I want to add:

- other kernels (thin plate spline, logarithmic)
- 1D Summation via KeOps as alternative to NFFT (could be useful in particular, when we are considering a low number of (relevant) Fourier features like, e.g., in the Gaussian kernel)
- for $d=2$ and $d=3$: add fast Fourier summation (without slicing)
