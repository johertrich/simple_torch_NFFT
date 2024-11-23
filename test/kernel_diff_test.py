import torch
from simple_torch_NFFT import Fastsum
from simple_torch_NFFT.fastsum.utils import get_median_distance
from simple_torch_NFFT.fastsum.functional import (
    fastsum_fft,
    fastsum_fft_precomputations,
    FastsumFFTAutograd,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

d = 10
kernel = "Laplace"

# number of Fourier coefficients to truncate,
# so far this value has to be chosen by hand,
# maybe I will add an adaptive selection for Gauss/Laplace/Matern at some point...
# higher value for rougher kernel and higher dimension
n_ft = 2048

# number of projections to test
P = 256

N = 1000
M = 1000

fastsum_naive = Fastsum(d, kernel=kernel, n_ft=n_ft, batched_autodiff=False)
fastsum = Fastsum(d, kernel=kernel, n_ft=n_ft)

x = torch.randn((N, d), device=device, dtype=torch.float)
y = torch.randn((M, d), device=device, dtype=torch.float)
x_weights = torch.rand(x.shape[0]).to(x)
output_sensitivities = torch.rand(y.shape[0]).to(y)

# choosing kernel parameter by median rule
med = get_median_distance(x, y)
scale = med

xis = fastsum.get_xis(P, device)

x.requires_grad_(True)
y.requires_grad_(True)
x_weights.requires_grad_(True)

s_sliced = fastsum_naive(x, y, x_weights, scale, xis)

loss = torch.sum(s_sliced * output_sensitivities)
x_grad, y_grad, x_weights_grad = torch.autograd.grad(loss, [x, y, x_weights])

s_sliced = fastsum(x, y, x_weights, scale, xis)

loss = torch.sum(s_sliced * output_sensitivities)
x_grad2, y_grad2, x_weights_grad2 = torch.autograd.grad(loss, [x, y, x_weights])

print("Difference in grad wrt x:", torch.sum((x_grad[:1] - x_grad2[:1]) ** 2).item())
print("Difference in grad wrt y:", torch.sum((y_grad[:1] - y_grad2[:1]) ** 2).item())
print(
    "Differnece in grad wrt x_weights:",
    torch.sum((x_weights_grad[:1] - x_weights_grad2[:1]) ** 2).item(),
)
