import torch
from simple_torch_NFFT import Fastsum
from simple_torch_NFFT.fastsum.utils import get_median_distance

device = "cuda" if torch.cuda.is_available() else "cpu"
torch._dynamo.config.cache_size_limit = 1024

d = 2
kernel = "Riesz"
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


N = 1000
M = 1000


x = torch.randn((N, d), device=device, dtype=torch.float)
y = torch.randn((M, d), device=device, dtype=torch.float)
x_weights = torch.ones(x.shape[0]).to(x)

# choosing kernel parameter by median rule
med = get_median_distance(x, y)
scale = med

fastsum = Fastsum(
    d,
    kernel=kernel,
    n_ft=n_ft,
    kernel_params=kernel_params,
    slicing_mode="non-sliced",
    batched_autodiff=False,
)

# compute naive kernel sum
s_naive = fastsum.naive(x, y, x_weights, scale)
# compute fastsum
s_fastsum = fastsum(x, y, x_weights, scale)
error = torch.sum(torch.abs(s_fastsum - s_naive)) / torch.sum(torch.abs(s_naive))
print(f"Relative L1-error: {error.item()}")
