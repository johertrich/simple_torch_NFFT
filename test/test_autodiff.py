from simple_torch_NFFT import NFFT, GaussWindow
from simple_torch_NFFT.nfft import ndft_adjoint, ndft_forward
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64
m = 4
sigma = 2

N = (2**10,)
J = 20000
batch_x = 2
batch_f = 2

nfft = NFFT(N, m=m, sigma=sigma, device=device, double_precision=double_precision)

x = (
    torch.rand(
        (batch_x, 1, J, len(N)),
        device=device,
        dtype=float_type,
    )
    - 0.5
)

f = torch.randn((batch_x, batch_f, J), dtype=complex_type, device=device)
f.requires_grad_(True)

# compute NFFT
fHat = nfft.adjoint(x, f)
loss = torch.sum(torch.abs(fHat))

f_grad = torch.autograd.grad(loss, f)[0]


# comparison with NDFT
fHat_dft = torch.stack(
    [
        torch.stack([ndft_adjoint(x[i, 0], f[i, j], N) for j in range(f.shape[1])], 0)
        for i in range(x.shape[0])
    ],
    0,
)

loss_dft = torch.sum(torch.abs(fHat_dft))

f_grad2 = torch.autograd.grad(loss_dft, f)[0]

print(torch.mean(torch.abs(f_grad - f_grad2) ** 2))

fHat_shape = [batch_x, batch_f] + list(N)

# test data
fHat = torch.randn(fHat_shape, dtype=complex_type, device=device)
fHat.requires_grad_(True)

# compute NFFT
f = nfft(x, fHat)
loss = torch.sum(torch.abs(f))
fHat_grad = torch.autograd.grad(loss, fHat)[0]

# comparison with NDFT
f_dft = torch.stack(
    [
        torch.stack(
            [ndft_forward(x[i, 0], fHat[i, j]) for j in range(fHat.shape[1])],
            0,
        )
        for i in range(x.shape[0])
    ],
    0,
)
loss_dft = torch.sum(torch.abs(f_dft))
fHat_grad2 = torch.autograd.grad(loss_dft, fHat)[0]
print(torch.mean(torch.abs(fHat_grad - fHat_grad2) ** 2))
