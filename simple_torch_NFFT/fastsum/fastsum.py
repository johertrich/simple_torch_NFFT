import torch
from .basis_funs import (
    Gaussian_kernel_fun_ft,
    Matern_kernel_fun_ft,
    f_fun_ft,
    thin_plate_f,
    logarithmic_f,
)
from .functional import (
    fastsum_fft,
    fastsum_fft_precomputations,
    FastsumFFTAutograd,
    FastsumEnergyAutograd,
    fast_energy_summation,
)
from simple_torch_NFFT import NFFT
from .utils import (
    compute_sliced_factor,
    compute_thin_plate_constant,
    compute_logarithmic_constant,
)
import importlib.resources
import h5py
import numpy as np
import urllib.request
import os
import hashlib


class Fastsum(torch.nn.Module):
    def __init__(
        self,
        dim,
        kernel="Gauss",
        kernel_params=dict(),
        slicing_mode=None,
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
            assert (
                "nu" in kernel_params.keys()
            ), "For the Matern kenrel, the smoothness parameter nu must be contained in kernel_params"
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
        elif kernel == "thin_plate":
            C = compute_thin_plate_constant(self.dim)
            basis_f = lambda x, scale: thin_plate_f(x, scale, C, self.dim)
            self.fourier_fun = lambda x, scale: f_fun_ft(x, scale, basis_f)
        elif kernel == "logarithmic":
            C = compute_logarithmic_constant(self.dim)
            basis_f = lambda x, scale: logarithmic_f(x, scale, C)
            self.fourier_fun = lambda x, scale: f_fun_ft(x, scale, basis_f)
        elif kernel == "other":
            assert (
                "fourier_fun" in kernel_params.keys()
                or "basis_f" in kernel_params.keys()
            ), "For custom kernels the Fourier transform of the 1D kernel must be contained in kernel_params"
            if "fourier_fun" in kernel_params.keys():
                self.fourier_fun = kernel_params["fourier_fun"]
            else:
                self.fourier_fun = lambda x, scale: f_fun_ft(
                    x, scale, kernel_params["basis_f"]
                )
        else:
            raise NameError("Kernel not found!")

        if nfft is None and not self.energy_kernel:
            self.nfft = NFFT((n_ft,), m=2, device=device, no_compile=no_compile)
        else:
            self.nfft = nfft

        if slicing_mode is None:
            if self.dim in [3, 4]:
                slicing_mode = "spherical_design"
            elif self.dim <= 100:
                slicing_mode = "distance"
            else:
                slicing_mode = "orthogonal"

        assert slicing_mode in [
            "iid",
            "spherical_design",
            "orthogonal",
            "Sobol",
            "distance",
        ], "Unknown slicing mode"
        if slicing_mode == "spherical_design":
            if self.dim in [3, 4]:
                base_path = str(importlib.resources.files("simple_torch_NFFT"))
                self.xis_dict = {}
                self.P_list = []
                with h5py.File(
                    base_path + "/data/spherical_designs_S" + str(self.dim - 1) + ".h5",
                    "r",
                ) as f:
                    for P in f["xis"].keys():
                        self.P_list.append(int(P))
                        self.xis_dict[P] = torch.tensor(
                            f["xis"][P][()], dtype=torch.float, device=device
                        )
            elif self.dim <= 100:
                print(
                    "Spherical designs are only available for d=3 and d=4! Therefore distance slices are used!"
                )
                slicing_mode = "distance"
            else:
                print(
                    "Spherical designs are only available for d=3 and d=4! Therefore orthogonal slices are used!"
                )
                slicing_mode = "orthogonal"
        if slicing_mode == "Sobol":
            self.sobolEng = torch.quasirandom.SobolEngine(self.dim)
            eps = 1e-4
            self.gauss_quantile = lambda p: np.sqrt(2) * torch.erfinv(
                (2 - 2 * eps) * p - 1 + eps
            )
        if slicing_mode == "distance":
            if self.dim <= 100:
                fdir = "distance_directions"
                if not os.path.isdir(fdir):
                    os.mkdir(fdir)
                if self.dim <= 20:
                    fname = "distance_directions_dim_2_to_20.h5"
                    drive_id = "1vh9UxqXV2PcsUHDxIYyS55lgQTXqupHG"
                    md5_val = "da440f823168cd68bf79e26306580c7a"
                    # https://drive.google.com/file/d/1vh9UxqXV2PcsUHDxIYyS55lgQTXqupHG/view?usp=sharing
                elif self.dim <= 40:
                    fname = "distance_directions_dim_21_to_40.h5"
                    drive_id = "1Ah3WSn1TKR-4xGJp9bC95Y8Gbwr0VJOd"
                    md5_val = "3b3d27c02a300d829fbeb4e526111d6c"
                    # https://drive.google.com/file/d/1Ah3WSn1TKR-4xGJp9bC95Y8Gbwr0VJOd/view?usp=sharing
                elif self.dim <= 60:
                    fname = "distance_directions_dim_41_to_60.h5"
                    drive_id = "1XqReXjZ1q1mpGCrHwg4wUX4zTpNivB_T"
                    md5_val = "c52e03aca5e5d2e97a869de5b0ae4f56"
                    # https://drive.google.com/file/d/1XqReXjZ1q1mpGCrHwg4wUX4zTpNivB_T/view?usp=sharing
                elif self.dim <= 80:
                    fname = "distance_directions_dim_61_to_80.h5"
                    drive_id = "1lBWiDWg74nvzOm7_YtFgGZKH_ugiDSKC"
                    md5_val = "5cde0e44f2afae694c587db38504d4b2"
                    # https://drive.google.com/file/d/1lBWiDWg74nvzOm7_YtFgGZKH_ugiDSKC/view?usp=sharing
                elif self.dim <= 100:
                    fname = "distance_directions_dim_81_to_100.h5"
                    drive_id = "1rPmwIrBtWmGrcg37z_w5ILc2I9KR5ZYz"
                    md5_val = "65f138cd53f5b393cea3ba25c6bd4750"
                    # https://drive.google.com/file/d/1rPmwIrBtWmGrcg37z_w5ILc2I9KR5ZYz/view?usp=sharing
                if not os.path.isfile(fdir + "/" + fname):
                    print("Download precomputed distance slices...")
                    url = "https://drive.google.com/uc?export=download&id=" + drive_id
                    urllib.request.urlretrieve(url, fdir + "/" + fname)
                    print("Download completed...")
                # verify
                md5_file = hashlib.md5(
                    open(fdir + "/" + fname, "rb").read()
                ).hexdigest()
                assert (
                    md5_file == md5_val
                ), "Verification of the file with the distance slices failed!"
                self.xis_dict = {}
                self.P_list = []
                with h5py.File(fdir + "/" + fname, "r") as f:
                    for P in f[str(self.dim)].keys():
                        self.P_list.append(int(P))
                        self.xis_dict[P] = torch.tensor(
                            f[str(self.dim)][P][()], dtype=torch.float, device=device
                        )
            else:
                print(
                    "Precomputed distance slices are only available for d<=100! Therefore orthogonal slices are used!"
                )
                slicing_mode = "orthogonal"

        self.slicing_mode = slicing_mode
        self.batch_size_P = batch_size_P
        self.batch_size_nfft = batch_size_nfft
        self.x_range = x_range
        self.batched_autodiff = batched_autodiff

    def get_xis(self, P, device):
        if self.slicing_mode == "iid":
            xis = torch.randn(P, self.dim, device=device)
            xis = xis / torch.sqrt(torch.sum(xis**2, -1, keepdims=True))
        elif self.slicing_mode == "spherical_design" or self.slicing_mode == "distance":
            # There is not for any P a corresponding spherical design / set of saved distance slices.
            # Thus, we increase P to the next large value where a spherical design / set of saved distances slices is available.
            # If P is larger than the largest available spherical design / set of saved distance slices, we use that one.
            greater_P = [pp for pp in self.P_list if pp >= P]
            if len(greater_P) == 0:
                effP = np.max(self.P_list)
            else:
                effP = np.min(greater_P)
            xis = self.xis_dict[str(effP)]
            # random rotations for obtaining an unbiased estimator:
            rot = torch.randn((self.dim, self.dim), device=device)
            rot, _ = torch.linalg.qr(rot)
            xis = torch.matmul(xis, rot)
        elif self.slicing_mode == "orthogonal":
            rot = torch.randn((P // self.dim + 1, self.dim, self.dim), device=device)
            rot, _ = torch.linalg.qr(rot)
            rot = rot.reshape(-1, self.dim)
            xis = rot[:P]
        elif self.slicing_mode == "Sobol":
            self.sobolEng.reset()
            xis = self.gauss_quantile(
                self.sobolEng.draw(P).to(device).clip(1e-6, 1 - 1e-6)
            )
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
                out = (
                    FastsumEnergyAutograd.apply(
                        x, y, x_weights, self.sliced_factor, batch_size_P, xis
                    )
                    / scale
                )
            else:
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
