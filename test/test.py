from simple_torch_NFFT import NFFT
from simple_torch_NFFT.nfft import ndft_adjoint, ndft_forward
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64

N = 2**10
J = 20000
k = (
    torch.rand(
        (
            2,
            1,
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
nfft = NFFT(N, m=m, sigma=sigma, device=device, double_precision=double_precision)

#################################
###### Test adjoint... ##########
#################################

# test data
f = torch.randn((k.shape[0], 2, k.shape[2]), dtype=complex_type, device=device)

# compute NFFT
fHat = nfft.adjoint(k, f)

# comparison with NDFT
fHat_dft = torch.stack(
    [
        torch.stack(
            [ndft_adjoint(k[i, 0], f[i, j], ft_grid) for j in range(f.shape[1])], 0
        )
        for i in range(k.shape[0])
    ],
    0,
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
fHat = torch.randn((k.shape[0], 2, N), dtype=complex_type, device=device)

# compute NFFT
f = nfft(k, fHat)

# comparison with NDFT
f_dft = torch.stack(
    [
        torch.stack(
            [ndft_forward(k[i, 0], fHat[i, j], ft_grid) for j in range(fHat.shape[1])],
            0,
        )
        for i in range(k.shape[0])
    ],
    0,
)

# relative error
print(
    "Relativer Fehler",
    torch.sqrt(torch.sum(torch.abs(f - f_dft) ** 2) / torch.sum(torch.abs(f_dft) ** 2)),
)


##############################################
###### Compare with torch_nfft... ############
##############################################

try:
    import torch_nfft as tn

    torch_nfft_comparison = True
except:
    print("torch_nfft cannot be loaded. Omit time comparison")
    torch_nfft_comparison = False


def batched_nfft(nfft_fun, points, inp):
    torch.cat(
        [
            torch.cat(
                [
                    nfft_fun(points[i : i + 1], inp[i : i + 1, j : j + 1])
                    for j in range(inp.shape[1])
                ],
                1,
            )
            for i in range(points.shape[0])
        ],
        0,
    )


sync = (
    (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)
)

N = 2**10
f_batches = 10
x_batches = 10
nfft = NFFT(N, m=m, sigma=sigma, device=device, double_precision=double_precision)

J = 100000
k = (
    torch.rand(
        (
            x_batches,
            1,
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
f = torch.randn([k.shape[0], f_batches, k.shape[-1]], dtype=complex_type, device=device)

batch_size = 1

# compile
fHat_stacked = batched_nfft(nfft.adjoint, k, f)
fHat = nfft.adjoint(k, f)
f = nfft(k, fHat)
f_stacked = batched_nfft(nfft, k, fHat)

# ground truth via NDFT
# fHat_dft=ndft_adjoint(k.squeeze(),f.squeeze(),ft_grid)

print("\n\nAdjoint:\n")

sync()
tic = time.time()
for _ in range(runs):
    fHat_stacked = batched_nfft(nfft.adjoint, k, f)
    sync()
toc = time.time() - tic
print("Stacked:", toc)

# compute NFFT
sync()
tic = time.time()
for _ in range(runs):
    fHat = nfft.adjoint(k, f)
    sync()
toc = time.time() - tic
print("Simple:", toc)

if torch_nfft_comparison:
    k_tn = k.tile(1, f.shape[1], 1)
    batch_x = torch.arange(
        k_tn.shape[0] * k_tn.shape[1], device=device
    ).repeat_interleave(k_tn.shape[2])
    torch.cuda.synchronize()
    tic = time.time()
    for _ in range(runs):
        fHat_tn = tn.nfft_adjoint(
            f.flatten(), k_tn.flatten()[:, None], bandwidth=N, batch=batch_x, cutoff=m
        ).view(k_tn.shape[0], f.shape[1], -1)
        torch.cuda.synchronize()
    toc = time.time() - tic
    print("CUDA native", toc)


print("\n\nForward:\n")

# test data
fHat = torch.randn((k.shape[0], f.shape[1], N), dtype=complex_type, device=device)

sync()
tic = time.time()
for _ in range(runs):
    f_stacked = batched_nfft(nfft, k, fHat)
    sync()
toc = time.time() - tic
print("Stacked:", toc)

# compute NFFT
sync()
tic = time.time()
for _ in range(runs):
    f = nfft(k, fHat)
    sync()
toc = time.time() - tic
print("Simple:", toc)

if torch_nfft_comparison:
    k_tn = k.tile(1, f.shape[1], 1)
    batch_y = torch.arange(
        k_tn.shape[0] * k_tn.shape[1], device=device
    ).repeat_interleave(k_tn.shape[2])
    torch.cuda.synchronize()
    tic = time.time()
    for _ in range(runs):
        f_torch_nfft = tn.nfft_forward(
            fHat.view(-1, fHat.shape[2]),
            k_tn.flatten()[:, None],
            batch=batch_y,
            cutoff=m,
        ).view(k_tn.shape[0], fHat.shape[1], -1)
        torch.cuda.synchronize()
    toc = time.time() - tic
    print("CUDA native", toc)
