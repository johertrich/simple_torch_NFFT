import torch
from simple_torch_NFFT import Fastsum
from simple_torch_NFFT.fastsum.utils import get_median_distance
from simple_torch_NFFT.fastsum.functional import (
    fastsum_fft,
    fastsum_fft_precomputations,
)
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

d = 10
kernel = "Gauss"

# number of Fourier coefficients to truncate,
# so far this value has to be chosen by hand,
# maybe I will add an adaptive selection for Gauss/Laplace/Matern at some point...
# higher value for rougher kernel and higher dimension
n_ft = 2048

# number of projections to test
P = 256

N = 1000
M = 1000
batch = (2, 2)
torch.manual_seed(0)

fastsum_naive = Fastsum(d, kernel=kernel, n_ft=n_ft, batched_autodiff=False)
fastsum = Fastsum(d, kernel=kernel, n_ft=n_ft, slicing_mode="iid")

x = torch.randn(batch + (N, d), device=device, dtype=torch.float)
y = torch.randn(batch + (M, d), device=device, dtype=torch.float)
x_weights = torch.rand(x.shape[:-1]).to(x)
output_sensitivities = torch.rand(y.shape[:-1]).to(y)


# choosing kernel parameter by median rule
med = get_median_distance(x[0], y[0])
scale = med

xis = fastsum.get_xis(P, device)

x.requires_grad_(True)
y.requires_grad_(True)
x_weights.requires_grad_(True)

outs_outer = []
for b1 in range(batch[0]):
    outs_inner = []
    for b2 in range(batch[1]):
        outs_inner.append(
            fastsum_naive(x[b1, b2], y[b1, b2], x_weights[b1, b2], scale, xis)
        )
    outs_outer.append(torch.stack(outs_inner, 0))
s_naive = torch.stack(outs_outer, 0)

loss = torch.sum(s_naive * output_sensitivities)
x_grad, y_grad, x_weights_grad = torch.autograd.grad(loss, [x, y, x_weights])

s_sliced = fastsum(x, y, x_weights, scale, xis)

loss = torch.sum(s_sliced * output_sensitivities)
x_grad2, y_grad2, x_weights_grad2 = torch.autograd.grad(loss, [x, y, x_weights])

print("Norm of grad wrt x:", torch.sum((x_grad) ** 2).item())
print("Norm of grad wrt y:", torch.sum((y_grad) ** 2).item())
print("Norm of grad wrt x_weights:", torch.sum((x_weights_grad) ** 2).item())

print("Norm of grad wrt x:", torch.sum((x_grad2) ** 2).item())
print("Norm of grad wrt y:", torch.sum((y_grad2) ** 2).item())
print("Norm of grad wrt x_weights:", torch.sum((x_weights_grad2) ** 2).item())

print(
    "Difference in grad wrt x:",
    torch.sum((x_grad - x_grad2) ** 2).item() / torch.sum(x_grad**2).item(),
)
print(
    "Difference in grad wrt y:",
    torch.sum((y_grad - y_grad2) ** 2).item() / torch.sum(y_grad**2).item(),
)
print(
    "Differnece in grad wrt x_weights:",
    torch.sum((x_weights_grad - x_weights_grad2) ** 2).item()
    / torch.sum(x_weights_grad**2).item(),
)
