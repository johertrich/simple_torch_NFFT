import torch


class KaiserBesselWindow(torch.nn.Module):
    def __init__(
        self,
        n,
        N,
        m,
        sigma,
        device="cuda" if torch.cuda.is_available() else "cpu",
        float_type=torch.float32,
    ):
        # n: size of oversampled regular grid
        # N: size of not-oversampled regular grid
        # m: Window size
        # sigma: oversampling
        # method adapted from NFFT.jl
        super().__init__()
        self.n = torch.tensor(n, dtype=float_type, device=device)
        self.N = N
        self.m = m
        self.sigma = torch.tensor(sigma, dtype=float_type, device=device)
        inds = torch.cartesian_prod(
            *[
                torch.arange(
                    -self.N[i] // 2, self.N[i] // 2, dtype=float_type, device=device
                )
                for i in range(len(self.N))
            ]
        ).reshape(list(self.N) + [-1])
        self.ft = torch.prod(self.Fourier_coefficients(inds), -1)

    def forward(self, k):  # no check that abs(k)<m/n !
        b = (2 - 1 / self.sigma) * torch.pi
        arg = torch.sqrt(self.m**2 - self.n**2 * k**2)
        out = torch.sinh(b * arg) / (arg * torch.pi)
        out = torch.nan_to_num(
            out, nan=0.0
        )  # outside the range out has nan values... replace them by zero
        return torch.prod(out, -1)

    def Fourier_coefficients(self, inds):
        b = (2 - 1 / self.sigma) * torch.pi
        return torch.special.i0(
            self.m * torch.sqrt(b**2 - (2 * torch.pi * inds / self.n) ** 2)
        )


class GaussWindow(torch.nn.Module):
    def __init__(
        self,
        n,
        N,
        m,
        sigma,
        device="cuda" if torch.cuda.is_available() else "cpu",
        float_type=torch.float32,
    ):
        # n: size of oversampled regular grid
        # N: size of not-oversampled regular grid
        # m: Window size
        # sigma: oversampling
        super().__init__()
        self.n = torch.tensor(n, dtype=float_type, device=device)
        self.N = N
        self.m = m
        self.sigma = torch.tensor(sigma, dtype=float_type, device=device)
        inds = torch.cartesian_prod(
            *[
                torch.arange(
                    -self.N[i] // 2, self.N[i] // 2, dtype=float_type, device=device
                )
                for i in range(len(self.N))
            ]
        ).reshape(list(self.N) + [-1])
        self.ft = torch.prod(self.Fourier_coefficients(inds), -1)

    def forward(self, k):
        b = self.m / torch.pi
        out = 1 / (torch.pi * b) ** (0.5) * torch.exp(-((self.n * k) ** 2) / b)
        return torch.prod(out, -1)

    def Fourier_coefficients(self, inds):
        b = self.m / torch.pi
        return torch.exp(-((torch.pi * inds / self.n) ** 2) * b)
