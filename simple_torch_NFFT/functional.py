import torch
import numpy as np


def transposed_sparse_convolution(x, f, n, m, phi_conj, device):
    # x has batch dimension plus two more dimensions of size #basis points times dimension
    # f has batch dimension plus one more dimension of size #basis_points
    # n is a tuple of even values
    # phi_conj is function handle
    unpadded_size = torch.Size([np.prod([n[i] for i in range(len(n))])])
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

    # next term: +n//2 for index shift from -n/2 util n/2-1 to 0 until n-1, % to prevent overflows
    for i in range(len(n)):
        inds[..., i] = (inds[..., i] + n[i] // 2) % n[i]

    # linear indices for dimension axis
    cumprods = torch.cumprod(
        torch.flip(torch.tensor(n, dtype=torch.int, device=device), (0,)), 0
    )
    cumprods = cumprods[:-1]
    size_mults = torch.ones(len(n), dtype=torch.int, device=device)
    size_mults[1:] = cumprods
    size_mults = torch.flip(size_mults, (0,))  # cumulated sizes of the dimensions
    inds = torch.sum(inds * size_mults, -1)
    inds_tile = [
        increments.shape[i] // inds.shape[i] for i in range(len(increments.shape))
    ]
    inds = inds.tile(inds_tile)
    inds = inds.view(increments.shape[0], -1, increments.shape[-1])

    # handling batch dimensions in linear indexing
    inds = (
        inds
        + unpadded_size[0]
        * torch.arange(0, inds.shape[1], device=device, dtype=torch.int)[None, :, None]
    )

    # index operations
    g_linear = torch.zeros(
        unpadded_size[0] * inds.shape[1],
        device=device,
        dtype=increments.dtype,
    )
    g_linear.index_put_((inds.view(-1),), increments.view(-1), accumulate=True)
    g_shape = list(increments.shape[1:-1]) + [n[i] for i in range(len(n))]
    g = g_linear.view(g_shape)
    return g


def adjoint_nfft(x, f, N, n, m, phi_conj, phi_hat, device):
    # x has batch dimension plus two more dimensions of size #basis points times dimension
    # f has batch dimension plus one more dimension of size #basis_points
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
    # x has batch dimension plus two more dimensions of size #basis points times dimension
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

    # handling dimensions
    cumprods = torch.cumprod(
        torch.flip(torch.tensor(n, dtype=torch.int, device=device), (0,)), 0
    )
    nonpadded_size = cumprods[-1]
    cumprods = cumprods[:-1]
    size_mults = torch.ones(len(n), dtype=torch.int, device=device)
    size_mults[1:] = cumprods
    size_mults = torch.flip(size_mults, (0,))
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
    # x has batch dimension plus two more dimensions of size #basis points times dimension
    # f_hat has batch dimension plus d more dimensions of size N=(N_1,...,N_d)
    # n is tuple of even values
    # phi is function handle
    # N is a tuple of even values
    # phi_hat starts with negtive indices
    # f_hat starts negative indices
    g_hat = f_hat / phi_hat
    lastdims = [-i for i in range(len(n), 0, -1)]
    for i in lastdims:
        g_hat_shape = list(g_hat.shape)
        g_hat_shape[i] = (n[i] - N[i]) // 2
        pad = torch.zeros(g_hat_shape, device=device)
        g_hat = torch.cat((pad, g_hat, pad), i)
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
