import torch
from .basis_funs import (
    Gaussian_kernel_fun_ft,
    Matern_kernel_fun_ft,
    f_fun_ft,
    thin_plate_f,
    logarithmic_f,
    Riesz_f,
    Gauss_F,
    Laplace_F,
    energy_F,
    thin_plate_F,
    logarithmic_F,
    Riesz_F,
    Matern_F,
    Gauss_F_ft,
    Matern_F_ft,
    F_fun_ft,
)
from .functional import (
    sliced_fastsum_fft,
    fastsum_fft,
    fastsum_fft_precomputations,
    SlicedFastsumFFTAutograd,
    FastsumEnergyAutograd,
    fast_energy_summation,
)
from simple_torch_NFFT import NFFT
from .utils import (
    compute_sliced_factor,
    compute_thin_plate_constant,
    compute_logarithmic_constant,
    compute_Riesz_factor,
)
import importlib.resources
import h5py
import numpy as np
import urllib.request
import os
import hashlib

try:
    import pykeops

    no_keops = False
except:
    pykeops = ImportError("PyKeops is not installed.")
    no_keops = True


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
        self.non_sliced = False
        self.dim = dim
        self.basis_F = None

        if kernel == "Gauss":
            self.fourier_fun = lambda x, scale: Gaussian_kernel_fun_ft(
                x, self.dim, scale**2
            )
            self.basis_F = Gauss_F
            self.F_fourier_fun = lambda x, scale: Gauss_F_ft(x, self.dim, scale**2)
        elif kernel == "Matern":
            assert (
                "nu" in kernel_params.keys()
            ), "For the Matern kenrel, the smoothness parameter nu must be contained in kernel_params"
            nu = kernel_params["nu"]
            self.fourier_fun = lambda x, scale: Matern_kernel_fun_ft(
                x, self.dim, scale, nu
            )
            self.basis_F = lambda x, scale: Matern_F(x, scale, nu, device)
            self.F_fourier_fun = lambda x, scale: Matern_F_ft(x, self.dim, scale, nu)
        elif kernel == "Laplace":
            self.fourier_fun = lambda x, scale: Matern_kernel_fun_ft(
                x, self.dim, scale, 0.5
            )
            self.basis_F = Laplace_F
            self.F_fourier_fun = lambda x, scale: Matern_F_ft(x, self.dim, scale, 0.5)
        elif kernel == "energy":
            self.energy_kernel = True
            self.sliced_factor = compute_sliced_factor(self.dim)
            self.basis_F = energy_F
            G = lambda x, scale: self.basis_F(torch.sqrt(torch.sum(x**2, -1)), scale)
            self.F_fourier_fun = lambda x, scale: F_fun_ft(x, scale, G, self.dim)
        elif kernel == "thin_plate":
            C = compute_thin_plate_constant(self.dim)
            basis_f = lambda x, scale: thin_plate_f(x, scale, C, self.dim)
            self.fourier_fun = lambda x, scale: f_fun_ft(x, scale, basis_f)
            self.basis_F = thin_plate_F
            G = lambda x, scale: self.basis_F(torch.sqrt(torch.sum(x**2, -1)), scale)
            self.F_fourier_fun = lambda x, scale: F_fun_ft(x, scale, G, self.dim)
        elif kernel == "logarithmic":
            C = compute_logarithmic_constant(self.dim)
            basis_f = lambda x, scale: logarithmic_f(x, scale, C)
            self.fourier_fun = lambda x, scale: f_fun_ft(x, scale, basis_f)
            self.basis_F = logarithmic_F
            G = lambda x, scale: self.basis_F(torch.sqrt(torch.sum(x**2, -1)), scale)
            self.F_fourier_fun = lambda x, scale: F_fun_ft(x, scale, G, self.dim)
        elif kernel == "Riesz":
            r = kernel_params["r"]
            C = compute_Riesz_factor(self.dim, r)
            basis_f = lambda x, scale: Riesz_f(x, scale, r, C)
            self.fourier_fun = lambda x, scale: f_fun_ft(x, scale, basis_f)
            self.basis_F = lambda x, scale: Riesz_F(x, scale, r)
            G = lambda x, scale: self.basis_F(torch.sqrt(torch.sum(x**2, -1)), scale)
            self.F_fourier_fun = lambda x, scale: F_fun_ft(x, scale, G, self.dim)
        elif kernel == "other":
            if slicing_mode == "non-sliced":
                assert (
                    "F_fourier_fun" in kernel_params.keys()
                    or "basis_F" in kernel_params.keys()
                    or "G" in kernel_params.keys()
                ), "For custom kernels either the basis function F or the function G with K(x,y)=G(x-y) must be specified"
                if "F_fourier_fun" in kernel_params.keys():
                    self.F_fourier_fun = kernel_params["F_fourier_fun"]
                else:
                    if "basis_F" in kernel_params.keys():
                        self.basis_f = kernel_params["basis_F"]
                        G = lambda x, scale: self.basis_F(
                            torch.sqrt(torch.sum(x**2, -1)), scale
                        )
                    else:
                        G = kernel_params["G"]
                    self.F_fourier_fun = lambda x, scale: F_fun_ft(
                        x, scale, G, self.dim
                    )
            else:
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
                if "basis_F" in kernel_params.keys():  # required for naive summation
                    self.basis_F = kernel_params["basis_F"]
                    G = lambda x, scale: self.basis_F(
                        torch.sqrt(torch.sum(x**2, -1)), scale
                    )
                elif "G" in kernel_params.keys():
                    G = kernel_params["G"]
        else:
            raise NameError("Kernel not found!")

        if slicing_mode is None:
            if self.dim in [1, 2]:
                slicing_mode = "non-sliced"
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
            "non-sliced",
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
            elif self.dim <= 100 and self.dim > 4:
                print(
                    "Spherical designs are only available for d=3 and d=4! Therefore distance slices are used!"
                )
                slicing_mode = "distance"
            elif self.dim > 100:
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
        if slicing_mode == "non-sliced":
            assert (
                kernel != "other"
            ), "Non-sliced fast Fourier summation is not implemented for custom kernels so far"
            self.non_sliced = True
            self.batched_autodiff = (
                False  # batched autodiff does not make sense in this context
            )

        if nfft is None and not (self.energy_kernel and not self.non_sliced):
            if self.non_sliced:
                self.nfft = NFFT(
                    tuple([n_ft] * self.dim), m=2, device=device, no_compile=no_compile
                )
            else:
                self.nfft = NFFT((n_ft,), m=2, device=device, no_compile=no_compile)
        else:
            self.nfft = nfft

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
            # In dimension 2, spherical designs are equispaced points on the (semi-)circle (for even integrands).
            if self.dim == 2:
                phi = (
                    (torch.arange(P, device=device) + torch.rand((1,), device=device))
                    * torch.pi
                    / P
                )
                xis = torch.stack((torch.cos(phi), torch.sin(phi)), dim=1)
            else:
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

    def naive(self, x, y, x_weights, scale):
        assert self.basis_F is not None, "Basis function of the kernel not given!"
        if not no_keops:
            y = pykeops.torch.LazyTensor(y[None, :, :])
            x = pykeops.torch.LazyTensor(x[:, None, :])
            x_weights = pykeops.torch.LazyTensor(x_weights[:, None], axis=0)
        else:
            y = y[None, :, :]
            x = x[:, None, :]
            x_weights = x_weights[:, None]
        distance_mat = ((x - y) ** 2).sum(-1).sqrt()
        kernel_mat = self.basis_F(distance_mat, scale)
        kernel_sum = (kernel_mat * x_weights).sum(0).squeeze()
        return kernel_sum

    def forward(self, x, y, x_weights, scale, xis_or_P=None):
        if not self.non_sliced:
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
            elif self.non_sliced:
                raise NotImplementedError()
            else:
                out = SlicedFastsumFFTAutograd.apply(
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
            if self.energy_kernel and not self.non_sliced:
                out = (
                    fast_energy_summation(
                        x, y, x_weights, self.sliced_factor, batch_size_P, xis
                    )
                    / scale
                )
            elif self.non_sliced:
                x, y, scale_factor = fastsum_fft_precomputations(
                    x, y, scale, self.x_range
                )
                h = torch.arange(
                    (-self.nfft.N[0] + 1) // 2,
                    (self.nfft.N[0] + 1) // 2,
                    device=x.device,
                )
                my_shape = [-1] + [1 for _ in range(1, self.dim)]
                h = h.view(my_shape)
                my_tile = [1] + [h.shape[0] for _ in range(1, self.dim)]
                h = h.tile(my_tile)
                my_mesh = [h.clone()]
                for i in range(1, self.dim):
                    my_mesh.append(h.transpose(0, i).clone())
                my_mesh = torch.stack(my_mesh, -1)
                kernel_ft = self.F_fourier_fun(my_mesh, scale * scale_factor)
                out = fastsum_fft(x, y, x_weights, kernel_ft, self.nfft)
                return out
            else:
                x, y, scale_factor = fastsum_fft_precomputations(
                    x, y, scale, self.x_range
                )
                h = torch.arange(
                    (-self.nfft.N[0] + 1) // 2,
                    (self.nfft.N[0] + 1) // 2,
                    device=x.device,
                )
                kernel_ft = self.fourier_fun(h, scale * scale_factor)
                out = sliced_fastsum_fft(
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
