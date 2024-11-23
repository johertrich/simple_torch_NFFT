import torch
from simple_torch_NFFT import Fastsum
from simple_torch_NFFT.fastsum.utils import get_median_distance
from simple_torch_NFFT.fastsum.functional import fastsum_fft

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.random.manual_seed(0)
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
s_sliced = fastsum(x, y, x_weights, scale, xis=xis)
loss = torch.sum(s_sliced * output_sensitivities)
x_grad = torch.autograd.grad(loss, x)[0]

x = x.requires_grad_(False)
der_sums = fastsum_fft(
    y,
    x,
    output_sensitivities,
    scale,
    fastsum.x_range,
    fastsum.fourier_fun,
    xis,
    fastsum.nfft,
    fastsum.batch_size_P,
    fastsum.batch_size_nfft,
    1,
    False,
)

x_grad2 = (
    torch.nn.functional.conv1d(
        xis.transpose(0, 1).flatten().reshape([1, 1, -1]),
        der_sums.transpose(0, 1).unsqueeze(1),
        stride=P,
    ).squeeze()
    / P
)
x_grad2 = x_grad2 * x_weights[:,None]
print(x_grad[:1] / x_grad2[:1])  # should be scale_factor
