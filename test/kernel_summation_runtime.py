import torch
from simple_torch_NFFT import Fastsum
from simple_torch_NFFT.fastsum.utils import get_median_distance
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch._dynamo.config.cache_size_limit = 1024

# for gpu testing
sync = (
    (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)
)

d = 4
kernel = "Gauss"
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
Ps = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

N = 1000
M = N

n_runs = 2


x = torch.randn((N, d), device=device, dtype=torch.float)
y = torch.randn((M, d), device=device, dtype=torch.float)
x_weights = torch.ones(x.shape[0]).to(x)

# choosing kernel parameter by median rule
med = get_median_distance(x, y)
scale = med

fastsum = Fastsum(d, kernel=kernel, kernel_params=kernel_params, batch_size_P=64)
fastsum_keops = Fastsum(
    d, kernel=kernel, kernel_params=kernel_params, batch_size_P=64, nfft="keops"
)

times_naive = []
for i in range(n_runs + 2):
    sync()
    time.sleep(0.5)
    tic = time.time()
    s_naive = fastsum.naive(x, y, x_weights, scale)
    sync()
    toc = time.time() - tic
    if i >= 2:
        times_naive.append(toc)
print("Naive evaluation takes {0:.2E} seconds".format(np.mean(times_naive)))

for P in Ps:
    times_slicing = []
    errors_slicing = []
    for i in range(n_runs + 2):
        sync()
        time.sleep(0.5)
        tic = time.time()
        s_slicing = fastsum(x, y, x_weights, scale, P)
        sync()
        toc = time.time() - tic
        if i >= 2:
            times_slicing.append(toc)
            error = torch.sum(torch.abs(s_slicing - s_naive)) / torch.sum(
                torch.abs(s_naive)
            )
            errors_slicing.append(error.item())
    print(
        "Sliced evaluation with {0} slices takes {1:.2E} seconds and has an relative L1-error of {2:.2E}".format(
            P, np.mean(times_slicing), np.mean(errors_slicing)
        )
    )


for P in Ps:
    times_slicing = []
    errors_slicing = []
    for i in range(n_runs + 2):
        sync()
        time.sleep(0.5)
        tic = time.time()
        s_slicing = fastsum_keops(x, y, x_weights, scale, P)
        sync()
        toc = time.time() - tic
        if i >= 2:
            times_slicing.append(toc)
            error = torch.sum(torch.abs(s_slicing - s_naive)) / torch.sum(
                torch.abs(s_naive)
            )
            errors_slicing.append(error.item())
    print(
        "Sliced evaluation with keops and {0} slices takes {1:.2E} seconds and has an relative L1-error of {2:.2E}".format(
            P, np.mean(times_slicing), np.mean(errors_slicing)
        )
    )
