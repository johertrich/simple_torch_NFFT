from simple_torch_NFFT import NFFT, GaussWindow
from simple_torch_NFFT.nfft import ndft_adjoint, ndft_forward
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import torch_nfft as tn

    if torch.cuda.is_available():
        torch_nfft_comparison = True
    else:
        print("cuda not available. Omit torch_nfft in the time comparison")
        torch_nfft_comparison = False
except:
    print("torch_nfft cannot be loaded. Omit torch_nfft in the time comparison")
    torch_nfft_comparison = False

try:
    import pyNFFT3

    nfft3_comparison = True
except:
    print(
        "pyNFFT3 cannot be loaded. Maybe it is not installed? Omit pyNFFT3 in the time comparison"
    )
    nfft3_comparison = False

try:
    import torchkbnufft as tkbn

    tkbn_comparison = True
except:
    print(
        "torchkbnufft cannot be loaded. Maybe it is not installed? Omit pyNFFT3 in the time comparison"
    )
    tkbn_comparison = False

double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64
m = 2
sigma = 2


def test(N, J, batch_x, batch_f):
    x = (
        torch.rand(
            (batch_x, 1, J, len(N)),
            device=device,
            dtype=float_type,
        )
        - 0.5
    )

    # for NDFT comparison
    # ft_grid = torch.arange(-N[0] // 2, N[0] // 2, dtype=float_type, device=device)

    # init nfft
    nfft = NFFT(N, m=m, sigma=sigma, device=device, double_precision=double_precision)

    if tkbn_comparison:
        # we use window size=2*m, torchkbnufft window size = numpoints, therefore we set numpoints=2*m
        tkbn_obj = tkbn.KbNufft(
            im_size=N, numpoints=2 * m, table_oversamp=2**18, device=device
        )
        tkbn_adj = tkbn.KbNufftAdjoint(
            im_size=N,
            numpoints=2 * m,
            table_oversamp=2**18,
            dtype=float_type,
            device=device,
        )
        x_kb = 2 * torch.pi * x.clone()
        x_kb = x_kb.squeeze(1)
        x_kb = x_kb.transpose(-2, -1)

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
        "Relative Error simple:",
        torch.sqrt(
            torch.sum(torch.abs(fHat - fHat_dft) ** 2)
            / torch.sum(torch.abs(fHat_dft) ** 2)
        ).item(),
    )

    if tkbn_comparison:
        # this should be in a different test file... This is only for runtime comparison
        fHat_kb = tkbn_adj(f, x_kb)
        print(
            "Relative Error torchkbnufft:",
            torch.sqrt(
                torch.sum(torch.abs(fHat_kb - fHat_dft) ** 2)
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
        "Relative Error simple:",
        torch.sqrt(
            torch.sum(torch.abs(f - f_dft) ** 2) / torch.sum(torch.abs(f_dft) ** 2)
        ).item(),
    )

    if tkbn_comparison:
        f_kb = tkbn_obj(fHat, x_kb)
        print(
            "Relative Error torchkbnufft:",
            torch.sqrt(
                torch.sum(torch.abs(f_kb - f_dft) ** 2)
                / torch.sum(torch.abs(f_dft) ** 2)
            ).item(),
        )


dim = 2
if dim == 1:
    ##############################
    print("Test for d=1")
    ##############################

    N = (2**10,)
    J = 20000
    batch_x = 2
    batch_f = 2

    test(N, J, batch_x, batch_f)

if dim == 2:
    ##############################
    print("Test for d=2")
    ##############################

    N = (2**5, 2**5)
    J = 20000
    batch_x = 2
    batch_f = 2

    test(N, J, batch_x, batch_f)


if dim == 3:
    ##############################
    print("Test for d=3")
    ##############################

    N = (2**4, 2**4, 2**4)
    J = 20000
    batch_x = 2
    batch_f = 2

    test(N, J, batch_x, batch_f)
