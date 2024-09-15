from simple_torch_NFFT import NFFT
from simple_torch_NFFT.nfft import ndft_adjoint,ndft_adjoint_1d, ndft_forward
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64

N = (2**5,2**6)
J = 20000
k = (
    torch.rand(
        (
            2,
            1,
            J,
            len(N)
        ),
        device=device,
        dtype=float_type,
    )
    - 0.5
)
m = 4
sigma = 2


# for NDFT comparison
#ft_grid = torch.arange(-N[0] // 2, N[0] // 2, dtype=float_type, device=device)

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
            [ndft_adjoint(k[i, 0], f[i, j], N) for j in range(f.shape[1])], 0
        )
        for i in range(k.shape[0])
    ],
    0,
)
#fHat_dft_1d= torch.stack(
#    [
#        torch.stack(
#            [ndft_adjoint_1d(k[i, 0].squeeze(), f[i, j], ft_grid) for j in range(f.shape[1])], 0
#        )
#        for i in range(k.shape[0])
#    ],
#    0,
#)
print(fHat.shape,fHat_dft.shape)

# relative error
print(
    "Relativer Fehler",
    torch.sqrt(
        torch.sum(torch.abs(fHat - fHat_dft) ** 2) / torch.sum(torch.abs(fHat_dft) ** 2)
    ),
)
exit()

#################################
###### Test forward... ##########
#################################

# test data
fHat = torch.randn((k.shape[0], 2, N), dtype=complex_type, device=device)

# compute NFFT
f = nfft(k.unsqueeze(-1), fHat)

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

