import torch
from .basis_funs import Gaussian_kernel_fun_ft, Matern_kernel_fun_ft
from .functional import fastsum_fft
from simple_torch_NFFT import NFFT


class Fastsum(torch.nn.Module):
    def __init__(
        self,
        dim,
        kernel="Gauss",
        kernel_params=dict(),
        n_ft=None,
        nfft=None,
        x_range=0.3,
        batch_size_P=None,
        batch_size_nfft=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        no_compile=False,
    ):
        super().__init__()
        if n_ft is None:
            # as perspective: add adaptive selection here...
            n_ft = 1024
        self.device = device
        if nfft is None:
            self.nfft = NFFT((n_ft,), m=2, device=device, no_compile=no_compile)
        else:
            self.nfft = nfft
        self.dim = dim
        if kernel == "Gauss":
            self.fourier_fun = lambda x, scale: Gaussian_kernel_fun_ft(
                x, self.dim, scale**2
            )
        elif kernel == "Matern":
            nu = kernel_params["nu"]
            self.fourier_fun = lambda x, scale: Matern_kernel_fun_ft(
                x, self.dim, scale, nu
            )
        elif kernel == "Laplace":
            self.fourier_fun = lambda x, scale: Matern_kernel_fun_ft(
                x, self.dim, scale, 0.5
            )
        elif kernel == "other":
            self.fourier_fun = kernel_params["fourier_fun"]

        self.batch_size_P = batch_size_P
        self.batch_size_nfft = batch_size_nfft
        self.x_range = x_range

    def get_xis(self, P, device):
        xis = torch.randn(P, self.dim, device=device)
        xis = xis / torch.sqrt(torch.sum(xis**2, -1, keepdims=True))
        return xis

    def forward(self, x, y, x_weights, scale, P):
        batch_size_P = P if self.batch_size_P is None else self.batch_size_P
        batch_size_nfft = (
            batch_size_P if self.batch_size_nfft is None else self.batch_size_nfft
        )
        if P < batch_size_P:
            batch_size_P = P
        else:
            P = batch_size_P * (((P - 1) // batch_size_P) + 1)
        xis = self.get_xis(P, x.device)
        return fastsum_fft(
            x,
            y,
            x_weights,
            scale,
            self.x_range,
            self.fourier_fun,
            xis,
            self.nfft,
            self.batch_size_P,
            self.batch_size_nfft,
        )
