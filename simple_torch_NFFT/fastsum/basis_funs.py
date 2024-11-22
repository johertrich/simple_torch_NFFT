import numpy as np
import torch
import scipy


def Gaussian_kernel_fun_ft(grid, d, sigma_sq):
    # implementation of the Fourier transform of the one-dimensional counterpart of the Gaussian kernel
    # for numerical stability computations are done in the log-space
    k = grid
    args = 2 * np.pi**2 * sigma_sq * k**2
    log_args = torch.log(args)
    log_args[args == 0] = 0
    factor = d * torch.pi * torch.sqrt(sigma_sq / 2)
    log_out = (
        log_args * 0.5 * (d - 1)
        - args
        - torch.lgamma(torch.tensor((d + 2) / 2, device=grid.device))
    )
    out = torch.exp(log_out)
    if d > 1:
        out[args == 0] = 0
    else:
        out[args == 0] = 1 / scipy.special.gamma((d + 2) / 2)
    return out * factor


def Matern_kernel_fun_ft(grid1d, d, beta, nu):
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
        out[args == 0] = 0.0
    else:
        out[args == 0] = torch.exp(log_factor)
    return out
