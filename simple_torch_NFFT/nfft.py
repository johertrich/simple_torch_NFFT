import torch
import numpy as np

# Very simple but vectorized version of the NFFT

def ndft_adjoint_1d(x, f, fts):
    # not vectorized adjoint NDFT for test purposes
    fourier_tensor = torch.exp(2j * torch.pi * fts[:, None] * x[None, :])
    y = torch.matmul(fourier_tensor, f[:, None])
    return y.squeeze()


def ndft_adjoint(x, f, N):
    inds=torch.cartesian_prod(*[torch.arange(-N[i] // 2, N[i] // 2, dtype=x.dtype, device=x.device) for i in range(len(N))]).view(-1,len(N))
    fourier_tensor=torch.exp(2j*torch.pi * torch.sum(inds[:,None,:] * x[None,:,:],-1))
    y=torch.matmul(fourier_tensor,f.view(-1,1))
    return y.view(N)

def ndft_forward(x, fHat, fts):
    # not vectorized forward NDFT for test purposes
    fourier_tensor = torch.exp(-2j * torch.pi * fts[None, :] * x[:, None])
    y = torch.matmul(fourier_tensor, fHat[:, None])
    return y.squeeze()


def transposed_sparse_convolution(x, f, n, m, phi_conj, device):
    # x is four-dimensional: batch_x times 1 times #basis points times dimension
    # f is three-dimesnional: (1 or batch_x) times batch_f times #basis_points
    # n is even
    # phi_conj is function handle
    window_shape = [-1] + [1 for _ in range(len(x.shape) - 1)] + [len(n)]
    window = torch.cartesian_prod(
        *[
            torch.arange(0, 2 * m, device=device, dtype=torch.long)
            for _ in range(len(n))
        ]
    ).view(window_shape)
    inds = (torch.ceil(x * torch.tensor(n, dtype=x.dtype, device=device)).long() - m)[
        None
    ] + window
    increments = (
            phi_conj(
                x[None]
                - inds.to(x.dtype) / torch.tensor(n, dtype=x.dtype, device=device)
            )
         * f[None]
    )

    cumprods = torch.cumprod(
        torch.flip(torch.tensor(n, dtype=torch.long, device=device) + 2*m, (0,)), 0
    )
    padded_size=cumprods[-1]
    cumprods=cumprods[:-1]
    size_mults = torch.ones(len(n), dtype=torch.long, device=device)
    size_mults[1:] = cumprods
    size_mults = torch.flip(size_mults, (0,))

    g_linear = torch.zeros(
        (increments.shape[1] * increments.shape[2] * padded_size,),
        device=device,
        dtype=increments.dtype,
    )
    # next term: +n//2 for index shift from -n/2 util n/2-1 to 0 until n-1, other part for linear indices
    # +m and 2*m to prevent overflows around 1/2=-1/2
    inds = inds + torch.tensor(
        [n[i] // 2 + m for i in range(len(n))], dtype=torch.long, device=device
    )


    # handling dimensions
    inds = torch.sum(inds * size_mults, -1)

    # handling batch dimensions in linear indexing
    inds = (
        inds.tile(1, 1, f.shape[1], 1)
        + padded_size
        * torch.arange(0, increments.shape[2], device=device, dtype=torch.long)[
            None, None, :, None
        ]
        + padded_size
        * increments.shape[2]
        * torch.arange(0, increments.shape[1], device=device, dtype=torch.long)[
            None, :, None, None
        ]
    )

    g_linear.index_put_((inds.reshape(-1),), increments.reshape(-1), accumulate=True)
    g_shape = [x.shape[0], f.shape[1]] + [n[i] + 2 * m for i in range(len(n))]
    g = g_linear.view(g_shape)
    # handle overflows
    for i in range(len(n)):
        g.index_add_(
            i + 2,
            torch.arange(n[i], n[i] + m, dtype=torch.long, device=device),
            torch.narrow(g, i + 2, 0, m).clone(),
        )
        g.index_add_(
            i + 2,
            torch.arange(m, 2 * m, dtype=torch.long, device=device),
            torch.narrow(g, i + 2, n[i] + m, m).clone(),
        )
        g = torch.narrow(g, i + 2, m, n[i])
    return g


#@torch.compile
def adjoint_nfft(x, f, N, n, m, phi_conj, phi_hat, device):
    # x is two-dimensional: batch_dim times basis points
    # f has same size as x or is broadcastable
    # n is even
    # phi_conj is function handle
    # N is even
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
    # x is two-dimensional: batch_dim times basis points
    # g lives on (discretized) [-1/2,1/2)
    # n is even
    # phi is function handle
    window_shape = [-1] + [1 for _ in range(len(x.shape) - 1)] + [len(n)]
    window = torch.cartesian_prod(
        *[
            torch.arange(0, 2 * m, device=device, dtype=torch.long)
            for _ in range(len(n))
        ]
    ).view(window_shape)
    inds = (torch.ceil(x * torch.tensor(n, dtype=x.dtype, device=device)).long() - m)[
        None
    ] + window
    increments = phi(
                x[None]
                - inds.to(x.dtype) / torch.tensor(n, dtype=x.dtype, device=device)
            )
    # % n to prevent overflows
    for i in range(len(n)):
        inds[...,i]=(inds[...,i]+n[i]//2)%n[i]
    inds=inds.tile(1,1,g.shape[1],1,1)

    cumprods = torch.cumprod(
        torch.flip(torch.tensor(n, dtype=torch.long, device=device), (0,)), 0
    )
    nonpadded_size=cumprods[-1]
    cumprods=cumprods[:-1]
    size_mults = torch.ones(len(n), dtype=torch.long, device=device)
    size_mults[1:] = cumprods
    size_mults = torch.flip(size_mults, (0,))
    # handling dimensions
    inds = torch.sum(inds * size_mults, -1)

    inds = (
        inds
        + nonpadded_size
        * torch.arange(0, g.shape[1], device=device, dtype=torch.long)[
            None, None, :, None
        ]
        + nonpadded_size
        * g.shape[1]
        * torch.arange(0, x.shape[0], device=device, dtype=torch.long)[
            None, :, None, None
        ]
    )
    g_l = g.view(-1)[inds]
    g_l *= increments
    f = torch.sum(g_l, 0)
    return f


@torch.compile
def forward_nfft(x, f_hat, N, n, m, phi, phi_hat, device):
    # x is three-dimensional: batch_x times 1 times #basis points
    # f_hat has size (batch_x,batch_f,N)
    # n is even
    # phi is function handle
    # N is even
    # phi_hat starts with negtive indices
    # f_hat fängt mit negativen indizes an
    g_hat = f_hat / phi_hat
    for i in range(len(n)):
        g_hat_shape=list(g_hat.shape)
        g_hat_shape[i+2]=(n[i]-N[i])//2
        pad = torch.zeros(g_hat_shape, device=device)
        g_hat = torch.cat((pad, g_hat, pad), i+2)
    lastdims = [-i for i in range(len(n), 0, -1)]
    g_hat=torch.fft.fftshift(g_hat, lastdims)
    if len(n) == 1:
        g = torch.fft.fft(g_hat, norm="backward")
    elif len(n) == 2:
        g = torch.fft.fft2(g_hat, norm="backward")
    else:
        g = torch.fft.fftn(g_hat, norm="backward", dim=lastdims)
    g=torch.fft.ifftshift(g,  lastdims)  # shift such that g lives again on [-1/2,1/2)
    f = sparse_convolution(x, g, n, m, x.shape[2], phi, device)
    # f has same size as x
    return f


# wrap autograd function around AdjointNFFT
class AdjointNFFT(torch.autograd.Function):
    @staticmethod
    def forward(x, f, N, n, m, phi_conj, phi_hat, device):
        return adjoint_nfft(x, f, N, n, m, phi_conj, phi_hat, device)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, _, N, n, m, phi_conj, phi_hat, device = inputs
        ctx.save_for_backward(x, N, n, m, phi_conj, phi_hat, device)

    @staticmethod
    def backward(ctx, grad_output):
        x, N, n, m, phi_conj, phi_hat, device = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            # call forward_nfft in the backward pass
            # assume that phi is real-valued (otherwise we would need a conjugate around the phi_conj here)
            grad_f = forward_nfft(x, grad_output, N, n, m, phi_conj, phi_hat, device)
        return None, grad_f, None, None, None, None, None, None


# wrwap autograd function around ForwardNFFT
class ForwardNFFT(torch.autograd.Function):
    @staticmethod
    def forward(x, f_hat, N, n, m, phi, phi_hat, device):
        return forward_nfft(x, f_hat, N, n, m, phi, phi_hat, device)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, _, N, n, m, phi, phi_hat, device = inputs
        ctx.save_for_backward(x, N, n, m, phi, phi_hat, device)

    @staticmethod
    def backward(ctx, grad_output):
        x, N, n, m, phi, phi_hat, device = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            # call adjoint_nfft in the backward pass
            # assume that phi is real-valued (otherwise we would need a conjugate around the phi here)
            grad_f_hat = adjoint_nfft(x, grad_output, N, n, m, phi, phi_hat, device)

        return None, grad_f_hat, None, None, None, None, None, None


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
        # n: number of oversampled Fourier coefficients
        # N: number of not-oversampled Fourier coefficients
        # m: Window size
        # sigma: oversampling
        # method adapted from NFFT.jl
        super().__init__()
        self.n = torch.tensor(n,dtype=float_type,device=device)
        self.N = N
        self.m = m
        self.sigma = torch.tensor(sigma,dtype=float_type,device=device)
        inds = torch.cartesian_prod(*[torch.arange(-self.N[i] // 2, self.N[i] // 2, dtype=float_type, device=device) for i in range(len(self.N))]).reshape(list(self.N)+[-1])
        self.ft = torch.prod(self.Fourier_coefficients(inds),-1)

    def forward(self, k):  # no check that abs(k)<m/n !
        b = (2 - 1 / self.sigma) * torch.pi
        arg = torch.sqrt(self.m**2 - self.n**2 * k**2)
        out = torch.sinh(b * arg) / (arg * torch.pi)
        out = torch.nan_to_num(
            out, nan=0.0
        )  # outside the range out has nan values... replace them by zero
        return torch.prod(out,-1)

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
        # n: number of oversampled Fourier coefficients
        # N: number of not-oversampled Fourier coefficients
        # m: Window size
        # sigma: oversampling
        super().__init__()
        self.n = n
        self.N = N
        self.m = m
        self.sigma = sigma
        inds = torch.arange(-self.N // 2, self.N // 2, dtype=float_type, device=device)
        self.ft = self.Fourier_coefficients(inds)

    def forward(self, k):
        b = self.m / torch.pi
        return 1 / (torch.pi * b) ** (0.5) * torch.exp(-((self.n * k) ** 2) / b)

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
    ):
        # N: number of not-oversampled Fourier coefficients
        # n: oversampled number of Fourier coefficients
        # sigma: oversampling
        # m: Window size
        super().__init__()
        if isinstance(N,int):
            self.N = (2 * (N // 2),)  # make N even
        else:
            self.N = tuple([2 * (N[i] // 2) for i in range(len(N))])
        if n is None:
            self.n = tuple([2 * (int(sigma * self.N[i]) // 2) for i in range(len(self.N))])  # make n even
        else:
            if isinstance(n,int):
                self.n = (2 * (n // 2),)  # make n even
            else:
                self.n = tuple([2 * (n[i] // 2) for i in range(len(n))])
        self.m = m
        self.device = device
        self.float_type = torch.float64 if double_precision else torch.float32
        self.complex_type = torch.complex128 if double_precision else torch.complex64
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
            self.window = window

    def forward(self, x, f_hat):
        return ForwardNFFT.apply(
            x, f_hat, self.N, self.n, self.m, self.window, self.window.ft, self.device
        )

    def adjoint(self, x, f):
        return AdjointNFFT.apply(
            x,
            f,
            self.N,
            self.n,
            self.m,
            self.window,
            self.window.ft,
            self.device,
        )
