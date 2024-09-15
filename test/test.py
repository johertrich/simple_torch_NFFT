from simple_torch_NFFT import NFFT
from simple_torch_NFFT.nfft import ndft_adjoint, ndft_forward
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64


def test(N, J, batch_x, batch_f):
    x = (
        torch.rand(
            (batch_x, 1, J, len(N)),
            device=device,
            dtype=float_type,
        )
        - 0.5
    )
    m = 4
    sigma = 2

    # for NDFT comparison
    # ft_grid = torch.arange(-N[0] // 2, N[0] // 2, dtype=float_type, device=device)

    # init nfft
    nfft = NFFT(N, m=m, sigma=sigma, device=device, double_precision=double_precision)

    #################################
    ###### Test adjoint... ##########
    #################################

    # test data
    f = torch.randn((batch_x, batch_f, J), dtype=complex_type, device=device)

    # compute NFFT
    fHat = nfft.adjoint(x, f)
    # comparison with NDFT
    fHat_dft = torch.stack(
        [
            torch.stack(
                [ndft_adjoint(x[i, 0], f[i, j], N) for j in range(f.shape[1])], 0
            )
            for i in range(x.shape[0])
        ],
        0,
    )

    # relative error
    print(
        "Relative Error",
        torch.sqrt(
            torch.sum(torch.abs(fHat - fHat_dft) ** 2)
            / torch.sum(torch.abs(fHat_dft) ** 2)
        ).item(),
    )

    #################################
    ###### Test forward... ##########
    #################################

    fHat_shape = [batch_x, batch_f] + list(N)

    # test data
    fHat = torch.randn(fHat_shape, dtype=complex_type, device=device)

    # compute NFFT
    f = nfft(x, fHat)

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

    # relative error
    print(
        "Relative Error",
        torch.sqrt(
            torch.sum(torch.abs(f - f_dft) ** 2) / torch.sum(torch.abs(f_dft) ** 2)
        ).item(),
    )


##############################
print("Test for d=1")
##############################

N = (2**10,)
J = 20000
batch_x = 2
batch_f = 2

test(N, J, batch_x, batch_f)

##############################
print("Test for d=2")
##############################

N = (2**6, 2**6)
J = 20000
batch_x = 2
batch_f = 2

test(N, J, batch_x, batch_f)

##############################
print("Test for d=3")
##############################

N = (2**4, 2**4, 2**4)
J = 20000
batch_x = 2
batch_f = 2

test(N, J, batch_x, batch_f)
