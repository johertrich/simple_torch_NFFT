# Specification of the Implemented Fastsum Functions and Classes

Back to: [Main Page](../../readme.md), [Overview of Fast Kernel Summation](overview.md)

We describe all attributes of the basic classes and functions from the fastsum-related objects.

## `simple_torch_NFFT.Fastsum`

The `Fastsum` class inherits from `torch.nn.Module`. A `Fastsum` object can be created with
```
from simple_torch_NFFT import Fastsum

fastsum = Fastsum(dim, kernel="Gauss", kernel_params=dict(), slicing_mode="iid", n_ft=None, nfft=None, x_range=0.3, batch_size_P=None, batch_size_nfft=None, device="cuda" if torch.cuda.is_available() else "cpu", no_compile=False, batched_autodiff=True)
```
with required argument

- `dim`: int. Dimension of the underlying space.

and optional keyword arguments

- `kernel="Gauss"`: string. Name of the used kernel. Possible choices are `"Gauss"`, `"Laplace"`, `"Matern"`, `"energy"`, `"thin_plate"` and `"logarithmic"`. See the [Overview](overview.md) page for a description of the kernels. For using a custom kernel, pass `"other"` and one of the following arguments in the `kernel_params` dictionary:
    - `kernel_params["fourier_fun"]`: must be a function handle of the Fourier transform of the one-dimensional kernel (i.e., the Fourier transform of $f$ in the [background page](background.md))
    - `kernel_params["basis_f"]`: must be a function handle with the basis function of the one-dimensional kernel (i.e., the function $f$ in the [background page](background.md))
- `kernel_params=dict()`: `dict`. An additional dictionary with kernel parameters. This does not contain the scale/bandwidth parameter, but additional parameters like the smoothness parameter `nu` for the Matern kernel.
- `slicing_mode=None`: string. Selection of the slicing directions $\xi$. The default value is `"spherical_design"` for $d\in\\{3,4\\}$, `"distance"` for $d \leq 100$ but $d\not\in\\{3,4\\}$ and `"orthogonal"` otherwise. Available choices are:
	-  `"iid"`: chooses the slices iid from the uniform distribution on the sphere.
	- `"orthogonal"`: Chooses the directions $\xi_p$ to be a random orthogonal system when $P\le d$. If $P>d$, we choose several independently drawn sets of random orthogonal systems. See [orthogonal random features](https://arxiv.org/abs/1610.09072) (in a random features setting) for details.
	- `"Sobol"`: Use the first $P$ entries of a Sobol sequence, transform them by the inverse cdf of a normal distribution and project it onto a sphere.
	- `"distance"`: maximizers of the pairwise distance $\mathcal E(\xi_1,...,\xi_P)=\sum_{p,q=1}^P \\|\xi_p-\xi_q\\|$ (in practice we use a symmetrized form of $\mathcal E$, see Section 4.1 of [this paper](https://arxiv.org/abs/2410.01316). These directions were precomputed for $d<=100$ and will be downloaded during the initialization of the `Fastsum` object. For $d>100$, the code goes back to orthogonal slices which coincide with distance slices for $P\leq d$.
	- `"spherical_design"`: uses spherical $t$-designs. This choice is only available for `dim==3` and `dim==4`, but in this case it is usually the most accurate choice.
- `n_ft=None`: int. Number of coefficients which are used to truncate the Fourier series of the kernel. `None` for automatic choice (which is currently 1024; on a long-term perspective it is planned to implement something adaptively to the dimension...)
- `nfft=None`: `simple_torch_NFFT.NFFT`. NFFT object which is used. `None` for creating a new one with standard parameters.
- `x_range=0.3`: float. Input data is rescaled onto the interval `[-x_range,x_range]` in order to use the Fourier transform on a compact interval.
- `batch_size_P=None`: int. Batching over the projections. Reduces memory load, but too small values might slow down the computations. `None` for no batching.
- `batch_size_nfft=None`: int. Batching for the NFFT computation over the projections. Reduces memory load, but too small values might slow down the computations. `None` for same value as `batch_size_P`.
- `device="cuda" if torch.cuda.is_available() else "cpu"`: "cpu" or "cuda". Device where all parameters will be initialized.
- `no_compile=False`: boolean. Set to True for not using `torch.compile` in the NFFT. This makes the execution significantly slower, but can be useful for debugging since one gets a useful stacktrace.
- `batched_autodiff=True`: boolean. If set to `True`, we define a custom backward pass which computes the derivative again as fast kernel summation (and therefore avoids tracing through all slices in the forward pass). This heavily reduces the required memory for automatic differentiation and might be faster. If the value is set to `False`, we do not define a custom backward pass (and hence use standard autodiff).

## `simple_torch_NFFT.fastsum.get_median_distance`

A small helper function for computing kernel parameters by the median rule. The function can be called by

```
from simple_torch_NFFT.fastsum import get_median_distance

median = get_median_distance(x, y, batch_size=1000)
```

It computes the median distance $\\|x_n-y_m\\|$ for the given input points `x` and `y`. Since this can be very expansive when the number of points is large, we first subsample `batch_size` points from `x` and `y` and compute the median distance based on this subset. More precisely, we have the following input arguments:

- `x`: `torch.Tensor`. First set of input points given as tensor of shape `(N,d)`
- `y`: `torch.Tensor`. Second set of input points given as tensor of shape `(M,d)`
- `batch_size=1000`: int. Number of points which are randomly selected.