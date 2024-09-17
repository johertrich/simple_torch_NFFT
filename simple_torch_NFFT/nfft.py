import torch
import numpy as np
import warnings
import sys

never_compile = False

if torch.__version__ < "2.4.0" and sys.version >= "3.12":
    warnings.warn(
        "You are using a PyTorch version older than 2.4.0! In PyTorch 2.3 (and older) torch.compile does not work with Python 3.12+. Consider to update PyTorch to 2.4 to get the best performance."
    )
    never_compile = True


def ndft_adjoint(x, f, N):
    # not vectorized adjoint NDFT for test purposes
    inds = torch.cartesian_prod(
        *[
            torch.arange(-N[i] // 2, N[i] // 2, dtype=x.dtype, device=x.device)
            for i in range(len(N))
        ]
    ).view(-1, len(N))
    fourier_tensor = torch.exp(
        2j * torch.pi * torch.sum(inds[:, None, :] * x[None, :, :], -1)
    )
    y = torch.matmul(fourier_tensor, f.view(-1, 1))
    return y.view(N)


def ndft_forward(x, fHat):
    N = fHat.shape
    # not vectorized adjoint NDFT for test purposes
    inds = torch.cartesian_prod(
        *[
            torch.arange(-N[i] // 2, N[i] // 2, dtype=x.dtype, device=x.device)
            for i in range(len(N))
        ]
    ).view(-1, len(N))
    fourier_tensor = torch.exp(
        -2j * torch.pi * torch.sum(inds[None, :, :] * x[:, None, :], -1)
    )
    y = torch.matmul(fourier_tensor, fHat.view(-1, 1))
    return y.view(x.shape[0])


def transposed_sparse_convolution(x, f, n, m, phi_conj, device):
    # x is four-dimensional: batch_x times 1 times #basis points times dimension
    # f is three-dimesnional: (1 or batch_x) times batch_f times #basis_points
    # n is a tuple of even values
    # phi_conj is function handle
    padded_size = torch.Size([np.prod([n[i] + 2 * m for i in range(len(n))])])
    window_shape = [-1] + [1 for _ in range(len(x.shape) - 1)] + [len(n)]
    window = torch.cartesian_prod(
        *[torch.arange(0, 2 * m, device=device, dtype=torch.int) for _ in range(len(n))]
    ).view(window_shape)
    inds = (torch.ceil(x * torch.tensor(n, dtype=x.dtype, device=device)).int() - m)[
        None
    ] + window
    increments = (
        phi_conj(
            x[None] - inds.to(x.dtype) / torch.tensor(n, dtype=x.dtype, device=device)
        )
        * f[None]
    )

    cumprods = torch.cumprod(
        torch.flip(torch.tensor(n, dtype=torch.int, device=device) + 2 * m, (0,)), 0
    )
    cumprods = cumprods[:-1]
    size_mults = torch.ones(len(n), dtype=torch.int, device=device)
    size_mults[1:] = cumprods
    size_mults = torch.flip(size_mults, (0,))
    g_linear = torch.zeros(
        padded_size[0] * increments.shape[1] * increments.shape[2],
        device=device,
        dtype=increments.dtype,
    )
    # next term: +n//2 for index shift from -n/2 util n/2-1 to 0 until n-1, other part for linear indices
    # +m and 2*m to prevent overflows around 1/2=-1/2
    inds = inds + torch.tensor(
        [n[i] // 2 + m for i in range(len(n))], dtype=torch.int, device=device
    )

    # handling dimensions
    inds = torch.sum(inds * size_mults, -1)
    inds_tile = [
        increments.shape[i] // inds.shape[i] for i in range(len(increments.shape))
    ]
    inds = inds.tile(inds_tile)
    inds = inds.view(increments.shape[0], -1, increments.shape[-1])
    # handling batch dimensions in linear indexing
    inds = (
        inds
        + padded_size[0]
        * torch.arange(0, inds.shape[1], device=device, dtype=torch.int)[None, :, None]
    )

    g_linear.index_put_((inds.view(-1),), increments.view(-1), accumulate=True)
    g_shape = [x.shape[0], f.shape[1]] + [n[i] + 2 * m for i in range(len(n))]
    g = g_linear.view(g_shape)

    # handle overflows
    if len(n) <= 4:
        g[..., -2 * m : -m] += g[..., :m]
        g[..., m : 2 * m] += g[..., -m:]
        g = g[..., m:-m]
        if len(n) >= 2:
            g[..., -2 * m : -m, :] += g[..., :m, :]
            g[..., m : 2 * m, :] += g[..., -m:, :]
            g = g[..., m:-m, :]
        if len(n) == 3:
            g[..., -2 * m : -m, :, :] += g[..., :m, :, :]
            g[..., m : 2 * m, :, :] += g[..., -m:, :, :]
            g = g[..., m:-m, :, :]
        if len(n) == 4:
            g[..., -2 * m : -m, :, :, :] += g[..., :m, :, :, :]
            g[..., m : 2 * m, :, :, :] += g[..., -m:, :, :, :]
            g = g[..., m:-m, :, :, :]
    else:
        # if someone is crazy enough (and has time and resources) for trying an NFFT in >3 dimensions.
        # Currently throws errors with torch.compile, but eager execution works,
        # but since NFFTs in d>4 are likely to be intractable anyway, I won't invest effort to fix that...
        for i in range(len(n)):
            g.index_add_(
                len(g) - len(n) + i,
                torch.arange(n[i], n[i] + m, dtype=torch.int, device=device),
                torch.narrow(g, len(g) - len(n) + i, 0, m).clone(),
            )
            g.index_add_(
                len(g) - len(n) + i,
                torch.arange(m, 2 * m, dtype=torch.int, device=device),
                torch.narrow(g, len(g) - len(n) + i, n[i] + m, m).clone(),
            )
            g = torch.narrow(g, len(g) - len(n) + i, m, n[i])
    return g


def adjoint_nfft(x, f, N, n, m, phi_conj, phi_hat, device):
    # x is four-dimensional: batch_x times 1 times #basis points times dimension
    # f is three-dimesnional: (1 or batch_x) times batch_f times #basis_points
    # n is a tuple of even values
    # phi_conj is function handle
    # N is a tuple of even values
    # phi_hat starts with negative indices
    cut = [(n[i] - N[i]) // 2 for i in range(len(n))]
    g = transposed_sparse_convolution(x, f, n, m, phi_conj, device)
    lastdims = [-i for i in range(len(n), 0, -1)]
    g = torch.fft.ifftshift(g, lastdims)
    if len(n) == 1:
        g_hat = torch.fft.ifft(g, norm="forward")
    elif len(n) == 2:
        g_hat = torch.fft.ifft2(g, norm="forward")
    else:
        g_hat = torch.fft.ifftn(g, norm="forward", dim=lastdims)
    g_hat = torch.fft.fftshift(g_hat, lastdims)
    for i in range(len(n)):
        g_hat = torch.narrow(g_hat, len(g_hat.shape) - len(n) + i, cut[i], N[i])
    f_hat = g_hat / phi_hat
    # f_hat starts with negative indices
    return f_hat


def sparse_convolution(x, g, n, m, M, phi, device):
    # x is four-dimensional: batch_x times 1 times #basis points times dimension
    # g lives on (discretized) [-1/2,1/2)^d
    # n is a tuple of even values
    # phi is function handle
    window_shape = [-1] + [1 for _ in range(len(x.shape) - 1)] + [len(n)]
    window = torch.cartesian_prod(
        *[torch.arange(0, 2 * m, device=device, dtype=torch.int) for _ in range(len(n))]
    ).view(window_shape)
    inds = (torch.ceil(x * torch.tensor(n, dtype=x.dtype, device=device)).int() - m)[
        None
    ] + window
    increments = phi(
        x[None] - inds.to(x.dtype) / torch.tensor(n, dtype=x.dtype, device=device)
    )
    # % n to prevent overflows
    for i in range(len(n)):
        inds[..., i] = (inds[..., i] + n[i] // 2) % n[i]

    cumprods = torch.cumprod(
        torch.flip(torch.tensor(n, dtype=torch.int, device=device), (0,)), 0
    )
    nonpadded_size = cumprods[-1]
    cumprods = cumprods[:-1]
    size_mults = torch.ones(len(n), dtype=torch.int, device=device)
    size_mults[1:] = cumprods
    size_mults = torch.flip(size_mults, (0,))
    # handling dimensions
    inds = torch.sum(inds * size_mults, -1)
    # tiling
    inds_tile = (
        [1]
        + [
            max(inds.shape[i + 1], g.shape[i]) // inds.shape[i + 1]
            for i in range(len(g.shape) - len(n))
        ]
        + [1 for _ in range(len(inds.shape) - len(g.shape) + len(n) - 1)]
    )
    inds = inds.tile(inds_tile)

    inds_shape = inds.shape
    inds = inds.view(inds.shape[0], -1, inds.shape[-1])
    # handling batch dimensions in linear indexing
    inds = (
        inds
        + nonpadded_size
        * torch.arange(0, inds.shape[1], device=device, dtype=torch.int)[None, :, None]
    ).view(inds_shape)

    g_tile = [inds.shape[i + 1] // g.shape[i] for i in range(len(g.shape) - len(n))] + [
        1 for _ in range(len(n))
    ]
    g = g.tile(g_tile).view(-1)
    g_l = g[inds]
    g_l *= increments
    f = torch.sum(g_l, 0)
    return f


def forward_nfft(x, f_hat, N, n, m, phi, phi_hat, device):
    # x is four-dimensional: batch_x times 1 times #basis points times dimension
    # f_hat has size (batch_x,batch_f,N_1,...,N_d)
    # n is tuple of even values
    # phi is function handle
    # N is a tuple of even values
    # phi_hat starts with negtive indices
    # f_hat starts negative indices
    g_hat = f_hat / phi_hat
    for i in range(len(n)):
        g_hat_shape = list(g_hat.shape)
        g_hat_shape[i + 2] = (n[i] - N[i]) // 2
        pad = torch.zeros(g_hat_shape, device=device)
        g_hat = torch.cat((pad, g_hat, pad), i + 2)
    lastdims = [-i for i in range(len(n), 0, -1)]
    g_hat = torch.fft.fftshift(g_hat, lastdims)
    if len(n) == 1:
        g = torch.fft.fft(g_hat, norm="backward")
    elif len(n) == 2:
        g = torch.fft.fft2(g_hat, norm="backward")
    else:
        g = torch.fft.fftn(g_hat, norm="backward", dim=lastdims)
    g = torch.fft.ifftshift(g, lastdims)  # shift such that g lives again on [-1/2,1/2)
    f = sparse_convolution(x, g, n, m, x.shape[-2], phi, device)
    # f has same size as x (without last dimension)
    return f


# Autograd Wrapper for linear functions
class LinearAutograd(torch.autograd.Function):
    @staticmethod
    def forward(x, inp, forward, adjoint):
        return forward(x, inp)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, _, forward, adjoint = inputs
        ctx.adjoint = adjoint
        ctx.forward = forward
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            grad_inp = LinearAutograd.apply(x, grad_output, ctx.adjoint, ctx.forward)
        return None, grad_inp, None, None


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


class NFFT(torch.nn.Module):
    def __init__(
        self,
        N,
        m=4,
        n=None,
        sigma=2.0,
        window=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        double_precision=False,
        no_compile=False,
        grad_via_adjoint=True,
    ):
        # n: size of oversampled regular grid
        # N: size of not-oversampled regular grid
        # sigma: oversampling
        # m: Window size
        super().__init__()
        if isinstance(N, int):
            self.N = (2 * (N // 2),)  # make N even
        else:
            self.N = tuple([2 * (N[i] // 2) for i in range(len(N))])
        if n is None:
            self.n = tuple(
                [2 * (int(sigma * self.N[i]) // 2) for i in range(len(self.N))]
            )  # make n even
        else:
            if isinstance(n, int):
                self.n = (2 * (n // 2),)  # make n even
            else:
                self.n = tuple([2 * (n[i] // 2) for i in range(len(n))])
        self.m = m
        self.device = device
        self.float_type = torch.float64 if double_precision else torch.float32
        self.complex_type = torch.complex128 if double_precision else torch.complex64
        self.padded_size = int(
            np.prod([self.n[i] + 2 * self.m for i in range(len(self.n))])
        )
        if window is None:
            self.window = KaiserBesselWindow(
                self.n,
                self.N,
                self.m,
                tuple([self.n[i] / self.N[i] for i in range(len(self.N))]),
                device=device,
                float_type=self.float_type,
            )
        else:
            self.window = window(
                self.n,
                self.N,
                self.m,
                tuple([self.n[i] / self.N[i] for i in range(len(self.N))]),
                device=device,
                float_type=self.float_type,
            )
        if no_compile or never_compile:
            if never_compile:
                warnings.warn(
                    "Compile is deactivated since the PyTorch version is too old. Consider to update PyTorch to 2.4 or newer."
                )
            self.forward_fun = forward_nfft
            self.adjoint_fun = adjoint_nfft
        else:
            self.forward_fun = torch.compile(forward_nfft)
            self.adjoint_fun = torch.compile(adjoint_nfft)
        self.grad_via_adjoint = grad_via_adjoint

    def apply_forward(self, x, f_hat):
        return self.forward_fun(
            x, f_hat, self.N, self.n, self.m, self.window, self.window.ft, self.device
        )

    def apply_adjoint(self, x, f):
        return self.adjoint_fun(
            x, f, self.N, self.n, self.m, self.window, self.window.ft, self.device
        )

    def forward(self, x, f_hat):
        # check dimensions
        assert (
            f_hat.shape[-len(self.n) :] == self.window.ft.shape
        ), f"Shape {f_hat.shape} of f_hat does not match the size {self.N} of the regular grid!"
        assert (
            x.shape[1] == 1
        ), f"x needs to have size 1 at dimension 1, given shape was {x.shape}"
        assert (
            f_hat.shape[0] == 1 or f_hat.shape[0] == x.shape[0]
        ), f"f_hat needs to be broadcastable to x at dimension 0, given shapes were {f_hat.shape} (f_hat), {x.shape} (x)"
        assert len(x.shape) == 4 and len(f_hat.shape) == 2 + len(
            self.N
        ), f"x needs to be 4-dimensional and f_hat needs to have 2+dim dimensions, given shapes were {f_hat.shape} (f_hat), {x.shape} (x)"

        # apply NFFT
        if self.grad_via_adjoint:
            return LinearAutograd.apply(
                x, f_hat, self.apply_forward, self.apply_adjoint
            )
        else:
            return self.apply_forward(x, f_hat)

    def adjoint(self, x, f):
        # check dimensions
        assert (
            f.shape[-1] == x.shape[-2]
        ), f"Shape {x.shape} of basis points x does not match shape {f.shape} of f!"
        assert (
            x.shape[1] == 1
        ), f"x needs to have size 1 at dimension 1, given shape was {x.shape}"
        assert (
            f.shape[0] == 1 or f.shape[0] == x.shape[0]
        ), f"f needs to be broadcastable to x at dimension 0, given shapes were {f.shape} (f), {x.shape} (x)"
        assert (
            len(x.shape) == 4 and len(f.shape) == 3
        ), f"x needs to be 4-dimensional and f needs to be 3-dimensional, given shapes were {f.shape} (f), {x.shape} (x)"

        # apply NFFT
        if self.grad_via_adjoint:
            return LinearAutograd.apply(x, f, self.apply_adjoint, self.apply_forward)
        else:
            return self.apply_adjoint(x, f)
