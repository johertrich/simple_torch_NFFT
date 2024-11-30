# Specification of the Implemented NFFT Functions and Classes

Back to: [Main Page](../../readme.md), [Overview of the NFFT](overview.md)

We describe all attributes of the basic classes and functions from the NFFT-related objects.

## NFFT and NDFT

### `simple_torch_NFFT.NFFT`

The `simple_torch_NFFT.NFFT` object is a `torch.nn.Module`. 
It can be created by
```
from simple_torch_NFFT import NFFT

nfft = NFFT(N, m=4, n=None, sigma=2.0, window=None, device="cuda" if torch.cuda.is_available() else "cpu", double_precision=False, no_compile=False, grad_via_adjoint=True)
```
with required argument:

- `N`: tuple of integers with length of the dimension. Size of the regular grid.

and optional arguments:

- `m=4`: integer. Window size (at the end the window has size `2*m`)
- `n=None`: tuple of integers. Size of the oversampled grid; set to `None` to use an oversampling factor instead
- `sigma=2.0`: float. Oversampling factor; only used, when `n is None`.
- `window=None`: Constructor torch module. Set to `None` for a Kaiser Bessel window, set it to `simple_torch_NFFT.GaussWindow` for a Gauss window function.
- `device="cuda" if torch.cuda.is_available() else "cpu"`: `"cpu"` or `"cuda"`. Device where the Fourier coefficients of the window will be initialized.
- `double_precision=False`: boolean. Set to `True` to compute in double precision.
- `no_compile=False`: boolean. Set to `True` for not using `torch.compile`; this makes the execution significantly slower, but can be useful for debugging since one gets a useful stacktrace.
- `grad_via_adjoint=True`: boolean. Set to `False` for computing the backward pass in the autodiff module by differentiating through the forward computation instead of calling the adjoint.

The forward method is given by
```
f = nfft(x, f_hat) # coincides with nfft.forward(x, f_hat)
```
with requried arguments

- `x`: `torch.Tensor`. Basis points of the NFFT. It should have the shape `(batch_dims,M,d)`, where `M` the number of basis points and `d` is the dimension. `batch_dims` can be an arbitrary number of batch dimensions. The batch dimensions of `x` and `f_hat` have to be equal or broadcastable.
- `f_hat`: `torch.Tensor`. Function values of the regular grid. It should have the shape `(batch_dims,N_1,...,N_d)`, where `N_1,...,N_d` is the size of the regular grid, which was given as argument `N` to the constructor of `NFFT`. `batch_dims` can be an arbitrary number of batch dimensions. The batch dimensions of `x` and `f_hat` have to be equal or broadcastable.

and output

- `f`: `torch.Tensor`. Function values on the nonequspaced grid specified by the input `x`. It has size `(batch_dims,M)`, where `batch_dims` is the broadcasted size of the batch dimensions of the two input arguments.

Finally, the `NFFT` object has method
```
f_hat = nfft.adjoint(x, f)
```
with required arguments `x` and `f` and output `f_hat` which have the same shapes as for the forward method. Note that (in contrast to the equispaced Fourier transform), the adjoint NFFT is in general not the inverse of the NFFT.

### `simple_torch_NFFT.NDFT`

For test reasons we also implemented a non-equispaced DFT. We did not tune this implementations (so it is probably not computationally efficient). An NDFT object can be created by
```
from simple_torch_NFFT import NFFT

ndft = NDFT(N)
```
with required argument:

- `N`: tuple of integers with length of the dimension. Size of the regular grid.

It has the forward and adjoint methods
```
f = ndft(x, f_hat) # coincides with ndft.forward(x, f_hat)
f_hat = ndft.adjoint(x, f)
```
with the same arguments as the NFFT object.

## Window Functions

We implemented two different window functions, namely `simple_torch_NFFT.KaiserBesselWindow` and `simple_torch_NFFT.GaussWindow`. The implementation of these window functions is adapted from the [NFFT.jl](https://github.com/JuliaMath/NFFT.jl) library. The window function is specified in the NFFT function by passing the constructor. It takes the arguments

```
from simple_torch_NFFT import KaiserBesselWindow
window = KaiserBesselWindow(n, N, m, sigma, device="cuda" if torch.cuda.is_available() else "cpu")
```
where the arguments are the same as for the `NFFT` object. Since the constructor is not thought to be called in user-code it does not set default values (this is done in the `NFFT` constructor).

After the object is initialized it must have an attribute `window.ft` with the Fourier coefficients of the window function.