import torch
import numpy as np
import warnings
import sys
from .window_functions import KaiserBesselWindow
from .functional import forward_nfft, adjoint_nfft

never_compile = False

if torch.__version__ < "2.4.0" and sys.version >= "3.12":
    warnings.warn(
        "You are using a PyTorch version older than 2.4.0! In PyTorch 2.3 (and older) torch.compile does not work with Python 3.12+. Consider to update PyTorch to 2.4 to get the best performance."
    )
    never_compile = True


# Autograd Wrapper for the NFFT
class LinearAutograd(torch.autograd.Function):
    @staticmethod
    def forward(x, inp, forward, adjoint, is_forward):
        return forward(x, inp)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, inp, forward, adjoint, is_forward = inputs
        ctx.adjoint = adjoint
        ctx.forward = forward
        ctx.is_forward = is_forward
        ctx.save_for_backward(x, inp)

    @staticmethod
    def backward(ctx, grad_output):
        (x, inp) = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            grad_inp = LinearAutograd.apply(
                x, grad_output, ctx.adjoint, ctx.forward, not ctx.is_forward
            )
            collapse_dims = tuple(
                [i for i in range(len(inp.shape)) if inp.shape[i] == 1]
            )
            if len(collapse_dims) > 0:
                grad_inp = torch.sum(grad_inp, collapse_dims, keepdims=True)
        else:
            grad_inp = None
        if ctx.needs_input_grad[0]:
            d = x.shape[-1]
            f_hat_mod = inp if ctx.is_forward else grad_output
            mults = grad_output if ctx.is_forward else inp
            N = f_hat_mod.shape[-d:]
            add_shape = (
                [1 for _ in range(len(f_hat_mod.shape) - len(N))] + list(N) + [-1]
            )
            inds = torch.cartesian_prod(
                *[
                    torch.arange(-N[i] // 2, N[i] // 2, dtype=x.dtype, device=x.device)
                    for i in range(len(N))
                ]
            ).reshape(add_shape)
            perm = [len(add_shape) - 1] + list(range(len(add_shape) - 1))
            inds = inds.permute(perm)
            x_mod = x.unsqueeze(0)
            f_hat_mod = -2j * torch.pi * inds * f_hat_mod.unsqueeze(0)
            if ctx.is_forward:
                grad_x = LinearAutograd.apply(
                    x_mod, f_hat_mod, ctx.forward, ctx.adjoint, True
                )
            else:
                grad_x = LinearAutograd.apply(
                    x_mod, f_hat_mod, ctx.adjoint, ctx.forward, True
                )
            perm = list(range(1, len(grad_x.shape))) + [0]
            grad_x = grad_x.permute(perm).conj() * mults.unsqueeze(-1)
            collapse_dims = tuple([i for i in range(len(x.shape)) if x.shape[i] == 1])
            if len(collapse_dims) > 0:
                grad_x = torch.sum(grad_x, collapse_dims, keepdims=True)
            grad_x = torch.real(grad_x)
        else:
            grad_x = None
        return grad_x, grad_inp, None, None, None


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
            self.window = KaiserBesselWindow
        self.window = self.window(
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
            self.forward_fun = torch.compile(
                forward_nfft, dynamic=False, mode="max-autotune"
            )
            self.adjoint_fun = torch.compile(
                adjoint_nfft, dynamic=False, mode="max-autotune"
            )
        self.grad_via_adjoint = grad_via_adjoint

    def apply_forward(self, x, f_hat):
        return self.forward_fun(
            x, f_hat, self.N, self.n, self.m, self.window, self.window.ft, self.device
        ).clone()

    def apply_adjoint(self, x, f):
        return self.adjoint_fun(
            x, f, self.N, self.n, self.m, self.window, self.window.ft, self.device
        ).clone()

    def forward(self, x, f_hat):
        # check dimensions
        assert (
            f_hat.shape[-len(self.n) :] == self.window.ft.shape
        ), f"Shape {f_hat.shape} of f_hat does not match the size {self.N} of the regular grid!"
        assert len(f_hat.shape) - len(x.shape) == -2 + len(
            self.N
        ), f"x  and f_hat need to have the same number of batch dimensions, given shapes were {f_hat.shape} (f_hat), {x.shape} (x)"

        # apply NFFT
        if self.grad_via_adjoint:
            return LinearAutograd.apply(
                x, f_hat, self.apply_forward, self.apply_adjoint, True
            )
        else:
            return self.apply_forward(x, f_hat)

    def adjoint(self, x, f):
        # check dimensions
        assert (
            f.shape[-1] == x.shape[-2]
        ), f"Shape {x.shape} of basis points x does not match shape {f.shape} of f!"
        assert (
            len(x.shape) == len(f.shape) + 1
        ), f"x f needs to have the same number of batch dimensions, given shapes were {f.shape} (f), {x.shape} (x)"

        # apply NFFT
        if self.grad_via_adjoint:
            return LinearAutograd.apply(
                x, f, self.apply_adjoint, self.apply_forward, False
            )
        else:
            return self.apply_adjoint(x, f)
