from simple_torch_NFFT import NFFT, NDFT
import torch
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64
m = 4
sigma = 2


def rand_batch_shapes(max_len=3, max_size=5):
    length = int(np.floor(np.random.uniform() * (max_len + 1)))
    batch_dims_x = []
    batch_dims_f = []
    for _ in range(length):
        num = int(np.floor(np.random.uniform() * (max_size - 1) + 1))
        collapse = np.random.uniform()
        if collapse < 1 / 3:
            batch_dims_x.append(1)
            batch_dims_f.append(num)
        elif collapse < 2 / 3:
            batch_dims_x.append(num)
            batch_dims_f.append(1)
        else:
            batch_dims_x.append(num)
            batch_dims_f.append(num)
    return batch_dims_x, batch_dims_f


def test(N, J, batch_dims_x, batch_dims_f):
    x_shape = batch_dims_x + [J, len(N)]
    x = (
        torch.rand(
            x_shape,
            device=device,
            dtype=float_type,
        )
        - 0.5
    )

    # for NDFT comparison
    # ft_grid = torch.arange(-N[0] // 2, N[0] // 2, dtype=float_type, device=device)

    # init nfft
    nfft = NFFT(N, m=m, sigma=sigma, device=device)
    ndft = NDFT(N)

    #################################
    ###### Test adjoint... ##########
    #################################

    # test data
    f_shape = batch_dims_f + [J]
    f = torch.randn(f_shape, dtype=complex_type, device=device)

    # compute NFFT
    fHat = nfft.adjoint(x, f)
    # comparison with NDFT
    fHat_dft = ndft.adjoint(x, f)

    # relative error
    rel_err = torch.sqrt(
        torch.sum(torch.abs(fHat - fHat_dft) ** 2) / torch.sum(torch.abs(fHat_dft) ** 2)
    ).item()

    print("Relative Error:", rel_err)

    #################################
    ###### Test forward... ##########
    #################################

    fHat_shape = batch_dims_f + list(N)

    # test data
    fHat = torch.randn(fHat_shape, dtype=complex_type, device=device)

    # compute NFFT
    f = nfft(x, fHat)

    # comparison with NDFT
    f_dft = ndft(x, fHat)

    # relative error
    rel_err = torch.sqrt(
        torch.sum(torch.abs(f - f_dft) ** 2) / torch.sum(torch.abs(f_dft) ** 2)
    ).item()
    print("Relative Error:", rel_err)


dim = 2
batch_x, batch_f = rand_batch_shapes()
if dim == 1:
    ##############################
    print("Test for d=1")
    ##############################

    N = (2**10,)
    J = 20000

    test(N, J, batch_x, batch_f)

if dim == 2:
    ##############################
    print("Test for d=2")
    ##############################

    N = (2**5, 2**5)
    J = 20000

    test(N, J, batch_x, batch_f)


if dim == 3:
    ##############################
    print("Test for d=3")
    ##############################

    N = (2**4, 2**4, 2**4)
    J = 20000

    test(N, J, batch_x, batch_f)
