import torch
from simple_torch_NFFT import Fastsum
from simple_torch_NFFT.fastsum.utils import get_median_distance

device = "cuda" if torch.cuda.is_available() else "cpu"

d = 10
kernel = "Laplace"

# number of Fourier coefficients to truncate,
# so far this value has to be chosen by hand,
# maybe I will add an adaptive selection for Gauss/Laplace/Matern at some point...
# higher value for rougher kernel and higher dimension
n_ft = 1024

# number of projections to test
Ps = [128, 256, 512]

N = 10000
M = 10000

fastsum = Fastsum(d, kernel=kernel, n_ft=n_ft)

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

s_naive = kernel_matrix @ x_weights

for P in Ps:
    s_sliced = fastsum(x, y, x_weights, scale, P)
    error = torch.sum(torch.abs(s_sliced - s_naive)) / torch.sum(torch.abs(s_naive))
    print(f"Relative L1-error for {P} slices: {error.item()}")
