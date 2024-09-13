from simple_torch_NFFT import NFFT
from simple_torch_NFFT.nfft import ndft_adjoint, ndft_forward
import torch
import time

import torch_nfft as tn

device = "cuda"
double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64

N = 2**10
J = 20000
k = (
    torch.rand(
        (
            2,
            J,
        ),
        device=device,
        dtype=float_type,
    )
    - 0.5
)
m = 4
sigma = 2

n = 2 * N

# for NDFT comparison
ft_grid = torch.arange(-N // 2, N // 2, dtype=float_type, device=device)

# init nfft
nfft = NFFT(N, m, sigma, device=device, double_precision=double_precision)

#################################
###### Test adjoint... ##########
#################################

# test data
f = torch.randn(k.shape, dtype=complex_type, device=device)

# compute NFFT
fHat = nfft.adjoint(k, f)

# comparison with NDFT
fHat_dft = torch.stack(
    [ndft_adjoint(k[i], f[i], ft_grid) for i in range(k.shape[0])], 0
)

# relative error
print(
    "Relativer Fehler",
    torch.sqrt(
        torch.sum(torch.abs(fHat - fHat_dft) ** 2) / torch.sum(torch.abs(fHat_dft) ** 2)
    ),
)


#################################
###### Test forward... ##########
#################################

# test data
fHat = torch.randn((k.shape[0], N), dtype=complex_type, device=device)

# compute NFFT
f = nfft(k, fHat)

# comparison with NDFT
f_dft = torch.stack(
    [ndft_forward(k[i], fHat[i], ft_grid) for i in range(k.shape[0])], 0
)

# relative error
print(
    "Relativer Fehler",
    torch.sqrt(torch.sum(torch.abs(f - f_dft) ** 2) / torch.sum(torch.abs(f_dft) ** 2)),
)


##############################################
###### Compare with torchkbnufft... ##########
##############################################
N = 2**10
nfft = NFFT(N, m, sigma, device=device, double_precision=double_precision)

J = 100000
k = (
    torch.rand(
        (
            100,
            J,
        ),
        device=device,
        dtype=float_type,
    )
    - 0.5
)
runs = 1
# k=k[None,:]

# test data
f = torch.randn(k.shape, dtype=complex_type, device=device)

batch_size = 1

# compile
fHat_stacked = torch.cat(
    [nfft.adjoint(k[i : i + 1], f[i : i + 1]) for i in range(k.shape[0])], 0
)
fHat_stacked = torch.cat(
    [
        nfft.adjoint(
            k[batch_size * i : batch_size * (i + 1)],
            f[batch_size * i : batch_size * (i + 1)],
        )
        for i in range(k.shape[0] // batch_size)
    ],
    0,
)
fHat = nfft.adjoint(k, f)
f = nfft(k, fHat)
f = torch.cat([nfft(k[i : i + 1], fHat[i : i + 1]) for i in range(k.shape[0])], 0)

# ground truth via NDFT
# fHat_dft=ndft_adjoint(k.squeeze(),f.squeeze(),ft_grid)

print("\n\nAdjoint:\n")

torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    fHat_stacked = torch.cat(
        [nfft.adjoint(k[i : i + 1], f[i : i + 1]) for i in range(k.shape[0])], 0
    )
    torch.cuda.synchronize()
toc = time.time() - tic
print("Stacked:", toc)

torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    fHat_stacked = torch.cat(
        [
            nfft.adjoint(
                k[batch_size * i : batch_size * (i + 1)],
                f[batch_size * i : batch_size * (i + 1)],
            )
            for i in range(k.shape[0] // batch_size)
        ],
        0,
    )
    torch.cuda.synchronize()
toc = time.time() - tic
print("Batched:", toc)

# compute NFFT
torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    fHat = nfft.adjoint(k, f)
    torch.cuda.synchronize()
toc = time.time() - tic
print("Simple:", toc)


batch_x = torch.arange(k.shape[0], device=device).repeat_interleave(k.shape[1])
torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    fHat_torch_nfft = tn.nfft_adjoint(
        f.flatten(), k.flatten()[:, None], bandwidth=N, batch=batch_x, cutoff=m
    )
    torch.cuda.synchronize()
toc = time.time() - tic
print("CUDA native", toc)

batch_x = torch.arange(k.shape[0], device=device).repeat_interleave(k.shape[1])
torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    fHat_torch_nfft = torch.cat(
        [
            tn.nfft_adjoint(f[i], k[i : i + 1].T, bandwidth=N, cutoff=m)
            for i in range(k.shape[0])
        ],
        0,
    )
    torch.cuda.synchronize()
toc = time.time() - tic
print("CUDA native stacked", toc)

print("\n\nForward:\n")

# test data
fHat = torch.randn((k.shape[0], N), dtype=complex_type, device=device)

torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    f = torch.cat([nfft(k[i : i + 1], fHat[i : i + 1]) for i in range(k.shape[0])], 0)
    torch.cuda.synchronize()
toc = time.time() - tic
print("Stacked:", toc)

# compute NFFT
torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    f = nfft(k, fHat)
    torch.cuda.synchronize()
toc = time.time() - tic
print("Simple:", toc)

batch_y = torch.arange(k.shape[0], device=device).repeat_interleave(k.shape[1])
torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    f_torch_nfft = tn.nfft_forward(fHat, k.flatten()[:, None], batch=batch_y, cutoff=m)
    torch.cuda.synchronize()
toc = time.time() - tic
print("CUDA native", toc)

batch_y = torch.arange(k.shape[0], device=device).repeat_interleave(k.shape[1])
torch.cuda.synchronize()
tic = time.time()
for _ in range(runs):
    f_torch_nfft = torch.cat(
        [
            tn.nfft_forward(fHat[i : i + 1], k[i : i + 1].T, cutoff=m)
            for i in range(k.shape[0])
        ],
        0,
    )
    torch.cuda.synchronize()
toc = time.time() - tic
print("CUDA native stacked", toc)
