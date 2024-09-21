from simple_torch_NFFT import NFFT, NDFT
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
batch_x = 1
batch_f = 1

nfft = NFFT(N, m=m, sigma=sigma, device=device, double_precision=double_precision, no_compile=True)
ndft = NDFT(N)

x = (
    torch.rand(
        (batch_x, 1, J, len(N)),
        device=device,
        dtype=float_type,
    )
    - 0.5
)

##############
# grad wrt x #
##############

# test data
fHat_shape = [batch_x, batch_f] + list(N)
fHat = torch.randn(fHat_shape, dtype=complex_type, device=device)
fHat=torch.real(fHat).to(complex_type)

x.requires_grad_(True)

# compute NFFT
f = nfft(x, fHat)
loss = torch.sum(torch.abs(f))
x_grad = torch.autograd.grad(loss, x)[0]
# comparison with NDFT
f_dft = ndft(x, fHat)

loss_dft = torch.sum(torch.abs(f_dft))
x_grad2 = torch.autograd.grad(loss_dft, x)[0]

print(torch.mean(torch.abs(x_grad - x_grad2) ** 2)/torch.mean(torch.abs(x_grad2) ** 2))

####################
# grad wrt f/f_hat #
####################

x.requires_grad_(False)
f = torch.randn((batch_x, batch_f, J), dtype=complex_type, device=device)
f.requires_grad_(True)

# compute NFFT
fHat = nfft.adjoint(x, f)
loss = torch.sum(torch.abs(fHat))

f_grad = torch.autograd.grad(loss, f)[0]


# comparison with NDFT
fHat_dft = ndft.adjoint(x, f)

loss_dft = torch.sum(torch.abs(fHat_dft))

f_grad2 = torch.autograd.grad(loss_dft, f)[0]

print(torch.mean(torch.abs(f_grad - f_grad2) ** 2)/torch.mean(torch.abs(f_grad2) ** 2))

fHat_shape = [batch_x, batch_f] + list(N)

# test data
fHat = torch.randn(fHat_shape, dtype=complex_type, device=device)
fHat.requires_grad_(True)

# compute NFFT
f = nfft(x, fHat)
loss = torch.sum(torch.abs(f))
fHat_grad = torch.autograd.grad(loss, fHat)[0]

# comparison with NDFT
f_dft = ndft(x, fHat)

loss_dft = torch.sum(torch.abs(f_dft))
fHat_grad2 = torch.autograd.grad(loss_dft, fHat)[0]
print(torch.mean(torch.abs(fHat_grad - fHat_grad2) ** 2)/torch.mean(torch.abs(fHat_grad2) ** 2))


