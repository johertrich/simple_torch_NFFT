import torch


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


def brute_force_resolve_batches(x, inp, method):
    if len(x.shape) == 2:
        return method(x, inp)
    if x.shape[0] > inp.shape[0]:
        return torch.stack(
            [
                brute_force_resolve_batches(x[i], inp[0], method)
                for i in range(x.shape[0])
            ],
            0,
        )
    if x.shape[0] < inp.shape[0]:
        return torch.stack(
            [
                brute_force_resolve_batches(x[0], inp[i], method)
                for i in range(inp.shape[0])
            ],
            0,
        )
    return torch.stack(
        [brute_force_resolve_batches(x[i], inp[i], method) for i in range(x.shape[0])],
        0,
    )


class NDFT(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def forward(self, x, f_hat):
        return brute_force_resolve_batches(x, f_hat, ndft_forward)

    def adjoint(self, x, f):
        return brute_force_resolve_batches(
            x, f, lambda x, f: ndft_adjoint(x, f, self.N)
        )
