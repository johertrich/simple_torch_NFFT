import torch
from simple_torch_NFFT import Fastsum
from simple_torch_NFFT.fastsum.utils import get_median_distance

device = "cuda" if torch.cuda.is_available() else "cpu"
torch._dynamo.config.cache_size_limit = 1024

d = 4
kernel = "Matern"
kernel_params = {}
if kernel == "Riesz":
    # choose exponent r for Riesz kernel
    kernel_params["r"] = 1.5
if kernel == "Matern":
    # choose smoothness parameter nu for Matern kernel
    kernel_params["nu"] = 3.5

# number of Fourier coefficients to truncate,
# so far this value has to be chosen by hand,
# maybe I will add an adaptive selection for Gauss/Laplace/Matern at some point...
# higher value for rougher kernel and higher dimension
n_ft = 1024
if kernel == "logarithmic":
    # the logarithmic kernel requires significantly more Fourier coefficients
    # than other kernels since K(x,y) -> -infty for x-y -> 0.
    n_ft = n_ft * 64

# number of projections to test
Ps = [128]

N = 1000
M = 900
batch = 3


x = torch.randn((batch, N, d), device=device, dtype=torch.float)
y = torch.randn((batch, M, d), device=device, dtype=torch.float)
x_weights = torch.ones((batch, N)).to(x)

# choosing kernel parameter by median rule
med = get_median_distance(x[0], y[0])
scale = med

# compute naive kernel sum
fastsum = Fastsum(d, kernel=kernel, kernel_params=kernel_params)
outs = []
for b in range(batch):
    outs.append(fastsum.naive(x[b], y[b], x_weights[b], scale))
s_naive = torch.stack(outs, 0)

slicing_mode = "iid"
fastsum = Fastsum(
    d,
    kernel=kernel,
    n_ft=n_ft,
    kernel_params=kernel_params,
    slicing_mode=slicing_mode,
)
P = 128
s_sliced = fastsum(x[0], y[0], x_weights[0], scale, P)
s_sliced = fastsum(x, y, x_weights, scale, P)
error = torch.sum(torch.abs(s_sliced - s_naive)) / torch.sum(torch.abs(s_naive))
print(f"Relative L1-error for {P} slices: {error.item()}")
