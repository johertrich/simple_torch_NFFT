import torch
from simple_torch_NFFT import Fastsum
from simple_torch_NFFT.fastsum.utils import get_median_distance

device = "cuda" if torch.cuda.is_available() else "cpu"
torch._dynamo.config.cache_size_limit = 1024

d = 4
kernel = "Riesz"
kernel_params = {}
if kernel == "Riesz":
    # choose exponent r for Riesz kernel
    kernel_params["r"] = 1.5
if kernel == "Matern":
    # choose smoothness parameter nu for Matern kernel
    kernel_params["nu"] = 1.5

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
Ps = [128, 256, 512, 1024, 2048]

N = 1000
M = 1000


x = torch.randn((N, d), device=device, dtype=torch.float)
y = torch.randn((M, d), device=device, dtype=torch.float)
x_weights = torch.ones(x.shape[0]).to(x)

# choosing kernel parameter by median rule
med = get_median_distance(x, y)
scale = med

# compute naive kernel sum:
if kernel == "Gauss":
    kernel_matrix = torch.exp(
        -0.5 * torch.sum((x[None, :, :] - y[:, None, :]) ** 2, -1) / scale**2
    )
elif kernel == "Laplace":
    kernel_matrix = torch.exp(
        -torch.sqrt(torch.sum((x[None, :, :] - y[:, None, :]) ** 2, -1)) / scale
    )
elif kernel == "energy":
    kernel_matrix = (
        -torch.sqrt(torch.sum((x[None, :, :] - y[:, None, :]) ** 2, -1)) / scale
    )
elif kernel == "thin_plate":
    dist_sq_matrix = torch.sum((x[None, :, :] - y[:, None, :]) ** 2, -1) / scale**2
    kernel_matrix = 0.5 * dist_sq_matrix * torch.log(dist_sq_matrix)
elif kernel == "logarithmic":
    dist_sq_matrix = torch.sum((x[None, :, :] - y[:, None, :]) ** 2, -1) / scale**2
    kernel_matrix = 0.5 * torch.log(dist_sq_matrix)
elif kernel == "Riesz":
    dist_sq_matrix = torch.sum((x[None, :, :] - y[:, None, :]) ** 2, -1) / scale**2
    kernel_matrix = -(dist_sq_matrix ** (kernel_params["r"] / 2))

s_naive = kernel_matrix @ x_weights

if d in [3, 4]:
    slicing_modes = ["iid", "orthogonal", "Sobol", "distance", "spherical_design"]
else:
    slicing_modes = ["iid", "orthogonal", "Sobol", "distance"]

for slicing_mode in slicing_modes:
    fastsum = Fastsum(
        d,
        kernel=kernel,
        n_ft=n_ft,
        kernel_params=kernel_params,
        batched_autodiff=False,
        slicing_mode=slicing_mode,
    )
    for P in Ps:
        s_sliced = fastsum(x, y, x_weights, scale, P)
        error = torch.sum(torch.abs(s_sliced - s_naive)) / torch.sum(torch.abs(s_naive))
        print(f"Relative L1-error for {P} slices: {error.item()}")
