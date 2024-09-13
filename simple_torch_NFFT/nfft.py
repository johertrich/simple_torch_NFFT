import torch

# Very simple but vectorized version of the NFFT

# Comments:
# - so far only 1D
# - so far only batching wrt the basis points
# - so far only autograd wrt f/f_hat not wrt basis points
# - oversampled and non-oversampled number of Fourier coefficients should be even
# - autograd not tested yet
# - does not like large cutoff paramters
#
# Other comments at functions in the code


def ndft_adjoint(x, f, fts):
    # not vectorized adjoint NDFT for test purposes
    fourier_tensor = torch.exp(2j * torch.pi * fts[:, None] * x[None, :])
    y = torch.matmul(fourier_tensor, f[:, None])
    return y.squeeze()


def ndft_forward(x, fHat, fts):
    # not vectorized forward NDFT for test purposes
    fourier_tensor = torch.exp(-2j * torch.pi * fts[None, :] * x[:, None])
    y = torch.matmul(fourier_tensor, fHat[:, None])
    return y.squeeze()


def transposed_sparse_convolution(x, f, n, m, phi_conj, device):
    # x is two-dimensional: batch_dim times basis points
    # f has same size as x or is broadcastable
    # n is even
    # phi_conj is function handle
    l = torch.arange(0, 2 * m, device=device, dtype=torch.long).view(2 * m, 1, 1)
    inds = (torch.ceil(n * x).long() - m)[None] + l
    increments = phi_conj(x[None, :, :] - inds.float() / n) * f
    g_linear = torch.zeros(
        (x.shape[0] * (n + 2 * m),), device=device, dtype=increments.dtype
    )
    # next term: +n//2 for index shift from -n/2 util n/2-1 to 0 until n-1, other part for linear indices
    # +m and 2*m to prevent overflows around 1/2=-1/2
    inds = (inds + n // 2 + m) + (n + 2 * m) * torch.arange(
        0, x.shape[0], device=device, dtype=torch.long
    )[None, :, None]
    g_linear.index_put_((inds.reshape(-1),), increments.reshape(-1), accumulate=True)
    g = g_linear.view(x.shape[0], n + 2 * m)
    # handle overflows
    g[:, -2 * m : -m] += g[:, :m]
    g[:, m : 2 * m] += g[:, -m:]
    return g[:, m:-m]


@torch.compile
def adjoint_nfft(x, f, N, n, m, phi_conj, phi_hat, device):
    # x is two-dimensional: batch_dim times basis points
    # f has same size as x or is broadcastable
    # n is even
    # phi_conj is function handle
    # N is even
    # phi_hat starts with negative indices
    cut = (n - N) // 2
    g = transposed_sparse_convolution(x, f, n, m, phi_conj, device)
    g = torch.fft.ifftshift(g, [-1])
    g_hat = torch.fft.ifft(g, norm="forward")
    g_hat = torch.fft.fftshift(g_hat, [-1])[:, cut:-cut]
    f_hat = g_hat / phi_hat
    # f_hat starts with negative indices
    return f_hat


def sparse_convolution(x, g, n, m, M, phi, device):
    # x is two-dimensional: batch_dim times basis points
    # g lives on (discretized) [-1/2,1/2)
    # n is even
    # phi is function handle
    l = torch.arange(0, 2 * m, device=device, dtype=torch.long).view(2 * m, 1, 1)
    inds = (torch.ceil(n * x).long() - m)[None, :, :] + l
    increments = phi(x[None, :, :] - inds / n).to(torch.complex64)
    inds = ((inds + n // 2) % n) + n * torch.arange(
        0, x.shape[0], device=device, dtype=torch.long
    )[
        :, None
    ]  # % n to prevent overflows
    # next term: +n//2 for index shift from -n/2 util n/2-1 to 0 until n-1, other part for linear indices
    g_l = g.view(-1)[inds].view(increments.shape)
    increments *= g_l
    f = torch.sum(increments, 0)
    return f


@torch.compile
def forward_nfft(x, f_hat, N, n, m, phi, phi_hat, device):
    # x is two-dimensional: batch_dim times basis points
    # f_hat has size (batch_size,N)
    # n is even
    # phi is function handle
    # N is even
    # phi_hat starts with negtive indices
    # f_hat fängt mit negativen indizes an
    g_hat = f_hat / phi_hat
    pad = torch.zeros((x.shape[0], (n - N) // 2), device=device)
    g_hat = torch.fft.fftshift(torch.cat((pad, g_hat, pad), 1), [-1])
    g = torch.fft.ifftshift(
        torch.fft.fft(g_hat, norm="backward"), [-1]
    )  # shift such that g lives again on [-1/2,1/2)
    f = sparse_convolution(x, g, n, m, x.shape[1], phi, device)
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
        self.n = n
        self.N = N
        self.m = m
        self.sigma = sigma
        inds = torch.arange(-self.N // 2, self.N // 2, dtype=float_type, device=device)
        self.ft = self.Fourier_coefficients(inds)

    def forward(self, k):  # no check that abs(k)<m/n !
        b = (2 - 1 / self.sigma) * torch.pi
        out = b / torch.pi * torch.ones_like(k)
        arg = torch.sqrt(self.m**2 - self.n**2 * k**2)
        out = torch.sinh(b * arg) / (arg * torch.pi)
        out = torch.nan_to_num(
            out, nan=0.0
        )  # outside the range out has nan values... replace them by zero
        return out

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
        m,
        sigma,
        window=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        double_precision=False,
    ):
        # N: number of not-oversampled Fourier coefficients
        # sigma: oversampling
        # m: Window size
        super().__init__()
        self.N = N
        self.n = int(sigma * N)
        self.m = m
        self.device = device
        self.float_type = torch.float64 if double_precision else torch.float32
        self.complex_type = torch.complex128 if double_precision else torch.complex64
        if window is None:
            self.window = KaiserBesselWindow(
                self.n,
                self.N,
                self.m,
                self.n / self.N,
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
            x, f, self.N, self.n, self.m, self.window, self.window.ft, self.device
        )
