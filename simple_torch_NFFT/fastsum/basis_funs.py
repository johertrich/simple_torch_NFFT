import numpy as np
import torch
import scipy


def Gaussian_kernel_fun_ft(grid, d, sigma_sq):
    # implementation of the Fourier transform of the one-dimensional counterpart of the Gaussian kernel
    # for numerical stability computations are done in the log-space
    k = grid
    args = 2 * np.pi**2 * sigma_sq * k**2
    log_args = torch.log(args)
    factor = d * torch.pi * torch.sqrt(sigma_sq / 2)
    log_out = (
        log_args * 0.5 * (d - 1)
        - args
        - torch.lgamma(torch.tensor((d + 2) / 2, device=grid.device))
    )
    out = torch.exp(log_out)
    if d > 1:
        out = torch.nan_to_num(out, nan=0.0)
    else:
        out[args == 0] = 1 / scipy.special.gamma((d + 2) / 2)
    return out * factor


def Gauss_F_ft(grid, d, sigma_sq):
    # implementation of the Fourier transform of the d-dimensional Gauss kernel
    return torch.exp(
        d / 2 * torch.log(2 * np.pi * sigma_sq)
        - 2 * np.pi**2 * sigma_sq * torch.sum(grid**2, -1)
    )


def Matern_kernel_fun_ft(grid1d, d, beta, nu):
    # implementation of the Fourier transform of the one-dimensional counterpart of the Matern kernel
    # for numerical stability computations are done in the log-space
    k = grid1d
    args = k**2
    log_args = torch.log(args)
    log_factor = (
        scipy.special.loggamma(nu + 0.5 * d)
        + d * np.log(np.pi)
        + d * torch.log(beta)
        + 0.5 * d * np.log(2.0)
        - scipy.special.loggamma(0.5 * d)
        - scipy.special.loggamma(nu)
        - 0.5 * d * np.log(nu)
    )
    log_out = 0.5 * (d - 1) * log_args - (nu + 0.5 * d) * torch.log(
        1 + (2 * torch.pi**2 * beta**2 / nu) * args
    )
    out = torch.exp(log_out + log_factor)
    if d > 1:
        out = torch.nan_to_num(out, nan=0.0)
    else:
        out[args == 0] = torch.exp(log_factor)
    return out


def Matern_F_ft(grid, d, beta, nu):
    log_out = (
        -(2 * nu + d)
        / 2
        * torch.log(1 + 2 * np.pi**2 * beta**2 * torch.sum(grid**2, -1) / nu)
    )
    log_constant = (
        scipy.special.loggamma((2 * nu + d) / 2)
        + d / 2 * np.log(2 * np.pi)
        + d * torch.log(beta)
        - scipy.special.loggamma(nu)
        - d / 2 * np.log(nu)
    )
    log_out = log_out + log_constant
    return torch.exp(log_out)


def f_fun_ft(grid1d, scale, f):
    # compute Fourier coefficients of some basis function f via the fft
    n_ft = grid1d.shape[0]
    vect = f(torch.abs(grid1d / n_ft), scale)
    vect_perm = torch.fft.ifftshift(vect)
    kernel_ft = 1 / n_ft * torch.fft.fftshift(torch.fft.fft(vect_perm))
    return kernel_ft


def F_fun_ft(grid, scale, G, d):
    # compute the Fourier coefficients of G where K(x,y)=G(x-y). If K is radial ,
    # then, G(x)=F(\|x\|), but generally this also works for other shift-invariant kernels
    n_ft = grid.shape[0]
    vect = G(grid / n_ft, scale)
    vect_perm = torch.fft.ifftshift(vect)
    if len(grid.shape) == 2:  # grid has d+1 dimensions
        vect_fft = torch.fft.fft(vect_perm)
    elif len(grid.shape) == 3:
        vect_fft = torch.fft.fft2(vect_perm)
    else:
        vect_fft = torch.fft.fftn(vect_perm)
    kernel_ft = 1 / (n_ft**d) * torch.fft.ifftshift(vect_fft)
    return kernel_ft


def thin_plate_f(x, scale, C, d):
    # basis function of the one-dimensional counterpart of the thin-plate spline kernel
    out = d * (x / scale) ** 2 * torch.log(x / scale) - C * (x / scale) ** 2
    out = torch.where(x == 0, torch.zeros_like(out), out)
    return out


def logarithmic_f(x, scale, C):
    # basis function of the one-dimensional counterpart of the logarithmic kernel
    out = torch.log(x / scale) - C
    out = torch.maximum(out, torch.tensor(-10.0, device=x.device, dtype=torch.float))
    return out


def Riesz_f(x, scale, r, C):
    return -C * x.abs() ** r / scale**r


def Gauss_F(x, scale):
    return (-0.5 / scale**2 * x**2).exp()


def Laplace_F(x, scale):
    return (-x.abs() / scale).exp()


def energy_F(x, scale):
    return -x.abs() / scale


def thin_plate_F(x, scale):
    out = (x / scale) ** 2 * (x.abs() / scale).log()
    if isinstance(out, torch.Tensor):
        out = torch.nan_to_num(out, nan=0.0)
    return out


def logarithmic_F(x, scale):
    out = (x.abs() / scale).log()
    if isinstance(out, torch.Tensor):
        out = torch.maximum(
            out, torch.tensor(-10.0, device=x.device, dtype=torch.float)
        )
    return out


def Riesz_F(x, scale, r):
    return -((x.abs() / scale) ** r)


def Matern_F(x, scale, nu, device):
    assert (
        (nu - 1 / 2) % 1
    ) == 0, (
        "Matern basis functions are only implmented for nu = p + 1/2 with integral p"
    )
    p = int(nu)
    p_ten = torch.tensor(p, dtype=torch.float, device=device)
    x_norm = x.abs() / scale
    summands = 0
    for n in range(p + 1):
        factor = torch.tensor(
            scipy.special.factorial(p + n)
            / (scipy.special.factorial(n) * scipy.special.factorial(p - n)),
            dtype=torch.float32,
            device=device,
        )
        add = factor * (4 * (2 * p + 1) * x_norm**2) ** ((p - n) / 2)
        summands = summands + add
    factor = torch.tensor(
        scipy.special.factorial(p) / scipy.special.factorial(2 * p),
        dtype=torch.float32,
        device=device,
    )
    prefactor = factor * (-torch.sqrt(2 * p_ten + 1) * x_norm).exp()
    return prefactor * summands
