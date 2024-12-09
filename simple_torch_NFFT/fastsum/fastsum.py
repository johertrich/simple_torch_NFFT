import torch
from .basis_funs import Gaussian_kernel_fun_ft, Matern_kernel_fun_ft
from .functional import (
    fastsum_fft,
    fastsum_fft_precomputations,
    FastsumFFTAutograd,
    fast_energy_summation,
)
from simple_torch_NFFT import NFFT
from .utils import compute_sliced_factor
import importlib.resources
import h5py
import numpy as np


class Fastsum(torch.nn.Module):
    def __init__(
        self,
        dim,
        kernel="Gauss",
        kernel_params=dict(),
        slicing_mode="iid",
        n_ft=None,
        nfft=None,
        x_range=0.3,
        batch_size_P=None,
        batch_size_nfft=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        no_compile=False,
        batched_autodiff=True,
    ):
        super().__init__()

        if n_ft is None:
            # as perspective: add adaptive selection here...
            n_ft = 1024

        self.device = device
        self.energy_kernel = False
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
        elif kernel == "energy":
            self.energy_kernel = True
            self.sliced_factor = compute_sliced_factor(self.dim)
        elif kernel == "other":
            self.fourier_fun = kernel_params["fourier_fun"]

        if nfft is None and not self.energy_kernel:
            self.nfft = NFFT((n_ft,), m=2, device=device, no_compile=no_compile)
        else:
            self.nfft = nfft

        assert slicing_mode in [
            "iid",
            "spherical_design",
            "orthogonal",
            "Sobol",
        ], "Unknown slicing mode"
        self.slicing_mode = slicing_mode
        if slicing_mode == "spherical_design":
            assert self.dim in [
                3,
                4,
            ], "Spherical designs are only available for d=3 and d=4"
            base_path = str(importlib.resources.files("simple_torch_NFFT"))
            self.xis_dict = {}
            self.P_list = []
            with h5py.File(
                base_path + "/data/spherical_designs_S" + str(self.dim - 1) + ".h5", "r"
            ) as f:
                for P in f["xis"].keys():
                    self.P_list.append(int(P))
                    self.xis_dict[P] = torch.tensor(
                        f["xis"][P][()], dtype=torch.float, device=device
                    )
        if slicing_mode == "Sobol":
            self.sobolEng = torch.quasirandom.SobolEngine(self.dim)
            self.gauss_quantile = lambda p: np.sqrt(2) * torch.erfinv(2 * p - 1)

        self.batch_size_P = batch_size_P
        self.batch_size_nfft = batch_size_nfft
        self.x_range = x_range
        self.batched_autodiff = batched_autodiff

    def get_xis(self, P, device):
        if self.slicing_mode == "iid":
            xis = torch.randn(P, self.dim, device=device)
            xis = xis / torch.sqrt(torch.sum(xis**2, -1, keepdims=True))
        elif self.slicing_mode == "spherical_design":
            # There is not for any P a corresponding spherical design.
            # Thus, we increase P to the next large value where a spherical design is available.
            # If P is larger than the largest available spherical design, we use that one.
            greater_P = [pp for pp in self.P_list if pp >= P]
            if len(greater_P) == 0:
                effP = np.max(self.P_list)
            else:
                effP = np.min(greater_P)
            xis = self.xis_dict[str(effP)]
        elif self.slicing_mode == "orthogonal":
            rot = torch.randn((P // self.dim + 1, self.dim, self.dim), device=device)
            rot, _ = torch.linalg.qr(rot)
            rot = rot.reshape(-1, self.dim)
            xis = rot[:P]
        elif self.slicing_mode == "Sobol":
            self.sobolEng.reset()
            xis = self.gauss_quantile(self.sobolEng.draw(P).to(device))
            zero_inds = torch.sqrt(torch.sum(xis**2, -1)) < 1e-6
            xis[zero_inds] = torch.randn_like(xis[zero_inds])
            xis = xis / torch.sqrt(torch.sum(xis**2, -1, keepdims=True))
        else:
            raise NotImplementedError("Unknown slicing mode!")
        return xis

    def forward(self, x, y, x_weights, scale, xis_or_P):
        if isinstance(xis_or_P, int):
            P = xis_or_P
            xis = None
        if isinstance(xis_or_P, torch.Tensor):
            xis = xis_or_P
            P = None
        if xis is None and P is None:
            raise ValueError(
                "either P (number of slices) or xis (Tensor containing slices) must be specified"
            )
        if xis is not None:
            P = xis.shape[0]
        batch_size_P = P if self.batch_size_P is None else self.batch_size_P
        batch_size_nfft = (
            batch_size_P if self.batch_size_nfft is None else self.batch_size_nfft
        )
        if P < batch_size_P:
            batch_size_P = P
        if batch_size_nfft > batch_size_P:
            batch_size_nfft = batch_size_P
        if xis is None:
            xis = self.get_xis(P, x.device)

        if self.batched_autodiff:
            if self.energy_kernel:
                raise NotImplementedError
            out = FastsumFFTAutograd.apply(
                x,
                y,
                x_weights,
                scale,
                self.nfft.N[0],
                self.x_range,
                self.fourier_fun,
                xis,
                self.nfft,
                self.batch_size_P,
                self.batch_size_nfft,
            )

        else:
            if self.energy_kernel:
                out = (
                    fast_energy_summation(
                        x, y, x_weights, self.sliced_factor, batch_size_P, xis
                    )
                    / scale
                )
            else:
                x, y, kernel_ft, h, _ = fastsum_fft_precomputations(
                    x, y, scale, self.x_range, self.fourier_fun, self.nfft.N[0]
                )
                out = fastsum_fft(
                    x,
                    y,
                    x_weights,
                    kernel_ft,
                    h,
                    xis,
                    self.nfft,
                    batch_size_P,
                    batch_size_nfft,
                    False,
                )
        return out
