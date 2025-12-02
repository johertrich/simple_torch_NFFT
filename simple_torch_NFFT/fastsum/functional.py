import torch


def fast_fourier_summation(x_proj, y_proj, x_weights, kernel_ft, nfft, take_sum):
    a = nfft.adjoint(-x_proj, x_weights.unsqueeze(0).unsqueeze(-2))
    a_time_kernel = a * kernel_ft
    if take_sum:
        return torch.sum(
            torch.real(nfft(-y_proj, a_time_kernel)).squeeze(-2), 0, keepdim=True
        )
    else:
        return torch.real(nfft(-y_proj, a_time_kernel)).squeeze(1)


def fastsum_fft_precomputations(x, y, scale, x_range):
    x_norm = torch.sqrt(torch.sum(x**2, -1)).reshape(-1)
    y_norm = torch.sqrt(torch.sum(y**2, -1)).reshape(-1)
    xy_norm = torch.cat((x_norm, y_norm), 0)
    max_norm = torch.max(xy_norm)

    scale_factor = 0.25 * x_range / max_norm
    scale_max = 0.1
    if scale * scale_factor > scale_max:
        scale_factor = scale_max / scale
    x = x * scale_factor
    y = y * scale_factor
    return x, y, scale_factor


class SlicedFastsumFFTAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        y,
        x_weights,
        scale,
        n_ft,
        x_range,
        fourier_fun,
        xis,
        nfft,
        batch_size_P=None,
        batch_size_nfft=None,
    ):
        x, y, scale_factor = fastsum_fft_precomputations(x, y, scale, x_range)
        h = torch.arange((-n_ft + 1) // 2, (n_ft + 1) // 2, device=x.device)
        kernel_ft = fourier_fun(h, scale * scale_factor)
        ctx.save_for_backward(x, y, x_weights, kernel_ft, h, xis, scale_factor)
        ctx.nfft = nfft
        ctx.batch_size_P = batch_size_P
        ctx.batch_size_nfft = batch_size_nfft
        return sliced_fastsum_fft(
            x,
            y,
            x_weights,
            kernel_ft,
            h,
            xis,
            nfft,
            batch_size_P,
            batch_size_nfft,
            derivative=False,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, y, x_weights, kernel_ft, h, xis, scale_factor = ctx.saved_tensors
        grad_x, grad_y, grad_x_weights = None, None, None
        if ctx.needs_input_grad[0]:
            grad_x = sliced_fastsum_fft(
                y,
                x,
                grad_output,
                kernel_ft,
                h,
                xis,
                ctx.nfft,
                ctx.batch_size_P,
                ctx.batch_size_nfft,
                derivative=True,
            )
            grad_x = grad_x * x_weights[:, None] * scale_factor
        if ctx.needs_input_grad[1]:
            grad_y = sliced_fastsum_fft(
                x,
                y,
                x_weights,
                kernel_ft,
                h,
                xis,
                ctx.nfft,
                ctx.batch_size_P,
                ctx.batch_size_nfft,
                derivative=True,
            )
            grad_y = grad_y * grad_output[:, None] * scale_factor
        if ctx.needs_input_grad[2]:
            grad_x_weights = sliced_fastsum_fft(
                y,
                x,
                grad_output,
                kernel_ft,
                h,
                xis,
                ctx.nfft,
                ctx.batch_size_P,
                ctx.batch_size_nfft,
                derivative=False,
            )
        return (
            grad_x,
            grad_y,
            grad_x_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fastsum_fft(x, y, x_weights, kernel_ft, nfft):
    a = nfft.adjoint(-x, x_weights)
    a_time_kernel = a * kernel_ft
    out = torch.real(nfft(-y, a_time_kernel))
    return out


def sliced_fastsum_fft(
    x,
    y,
    x_weights,
    kernel_ft,
    h,
    xis,
    nfft,
    batch_size_P=None,
    batch_size_nfft=None,
    derivative=False,
):
    P = xis.shape[0]
    M = y.shape[-2]
    batch_dims_y = list(y.shape[:-2])
    batch_dims_x = list(x.shape[:-2])
    if batch_size_P is None:
        batch_size_P = P
    if batch_size_nfft is None:
        batch_size_nfft = batch_size_P
    batch_size_nfft = min(batch_size_nfft, batch_size_P)

    if derivative:
        kernel_ft = kernel_ft * (2 * torch.pi * 1j * h) ** derivative

    xi = xis.view([P] + (len(batch_dims_y) + 1) * [1] + [xis.shape[-1]])

    def with_projections(xi):
        P_local = xi.shape[0]
        x_proj = (xi @ x.transpose(-1, -2)).reshape(
            [P_local] + batch_dims_x + [1, -1, 1]
        )
        y_proj = (xi @ y.transpose(-1, -2)).reshape(
            [P_local] + batch_dims_y + [1, -1, 1]
        )
        outs = torch.cat(
            [
                fast_fourier_summation(
                    x_proj[i * batch_size_nfft : (i + 1) * batch_size_nfft],
                    y_proj[i * batch_size_nfft : (i + 1) * batch_size_nfft],
                    x_weights,
                    kernel_ft,
                    nfft,
                    take_sum=not derivative,
                )
                for i in range(((P_local - 1) // batch_size_nfft) + 1)
            ],
            0,
        )
        print(outs.shape)

        if derivative:
            return (
                torch.nn.functional.conv1d(
                    xi.squeeze().transpose(0, 1).flatten().reshape([1, 1, -1]),
                    outs.transpose(0, 1).unsqueeze(1),
                    stride=P_local,
                )
                .squeeze()
                .unsqueeze(0)
            )
        else:
            return torch.sum(outs, 0, keepdim=True)

    outs = torch.cat(
        [
            with_projections(xi[i * batch_size_P : (i + 1) * batch_size_P])
            for i in range(((P - 1) // batch_size_P) + 1)
        ],
        0,
    )
    return torch.sum(outs, 0) / P


class FastsumEnergyAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, x_weights, sliced_factor, batch_size, xis):
        ctx.save_for_backward(x, y, x_weights, xis)
        ctx.sliced_factor = sliced_factor
        ctx.batch_size = batch_size
        return fast_energy_summation(x, y, x_weights, sliced_factor, batch_size, xis)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, x_weights, xis = ctx.saved_tensors
        grad_x, grad_y, grad_x_weights = None, None, None
        if ctx.needs_input_grad[0]:
            grad_x = fast_energy_summation_grad(
                y, x, grad_output, ctx.sliced_factor, ctx.batch_size, xis
            )
            grad_x = grad_x * x_weights[:, None]
        if ctx.needs_input_grad[1]:
            grad_y = fast_energy_summation_grad(
                x, y, x_weights, ctx.sliced_factor, ctx.batch_size, xis
            )
            grad_y = grad_y * grad_output[:, None]
        if ctx.needs_input_grad[2]:
            grad_x_weights = fast_energy_summation(
                y, x, grad_output, ctx.sliced_factor, ctx.batch_size, xis
            )
        return (
            grad_x,
            grad_y,
            grad_x_weights,
            None,
            None,
            None,
        )


def fastsum_energy_kernel_1D(x, x_weights, y):
    # Sorting algorithm for fast sumation with negative distance (energy) kernel
    N = x.shape[1]
    M = y.shape[1]
    P = x.shape[0]
    # Potential Energy
    sorted_yx, inds_yx = torch.sort(torch.cat((y, x), 1))
    inds_yx = inds_yx + torch.arange(P, device=x.device).unsqueeze(1) * (N + M)
    inds_yx = torch.flatten(inds_yx)
    weights_sorted = (
        torch.cat((torch.zeros_like(y), x_weights), 1).flatten()[inds_yx].reshape(P, -1)
    )
    pot0 = torch.sum(weights_sorted * (sorted_yx - sorted_yx[:, 0:1]), 1, keepdim=True)
    yx_diffs = sorted_yx[:, 1:] - sorted_yx[:, :-1]
    # Mults from cumsums shifted by 1
    mults_short = (
        torch.sum(x_weights, -1, keepdim=True)
        - 2 * torch.cumsum(weights_sorted, 1)[:, :-1]
    )
    mults = torch.zeros_like(weights_sorted)
    mults[:, 1:] = mults_short
    potential = torch.zeros_like(sorted_yx)
    potential[:, 1:] = yx_diffs.clone()
    potential = pot0 - torch.cumsum(potential * mults, 1)
    out1 = torch.zeros_like(sorted_yx).flatten()
    out1[inds_yx] = potential.flatten()
    out1 = out1.reshape(P, -1)
    out1 = out1[:, :M]
    return out1


def fast_energy_summation(x, y, x_weights, sliced_factor, batch_size, xis):
    # fast sum via slicing and sorting
    d = x.shape[1]
    P = xis.shape[0]

    def with_projections(xi):
        P_local = xi.shape[0]
        x_proj = (xi @ x.T).reshape(P_local, -1)
        y_proj = (xi @ y.T).reshape(P_local, -1)
        fastsum_energy = fastsum_energy_kernel_1D(
            x_proj, x_weights[None, :].tile(P_local, 1), y_proj
        )
        return sliced_factor * torch.sum(-fastsum_energy, 0)

    return (
        torch.sum(
            torch.stack(
                [
                    with_projections(xis[i * batch_size : (i + 1) * batch_size])
                    for i in range((P - 1) // batch_size + 1)
                ],
                0,
            ),
            0,
        )
        / P
    )


def fastsum_energy_kernel_1D_grad(x, x_weights, y):
    # Sorting algorithm for fast sumation with the gradient of the negative distance (energy) kernel
    N = x.shape[1]
    M = y.shape[1]
    P = x.shape[0]
    weights_sum = torch.sum(x_weights, 1, keepdim=True)
    # Potential Energy
    sorted_yx, inds_yx = torch.sort(torch.cat((y, x), 1))
    inds_yx = inds_yx + torch.arange(P, device=x.device).unsqueeze(1) * (N + M)
    inds_yx = torch.flatten(inds_yx)
    weights_sorted = (
        torch.cat((torch.zeros_like(y), x_weights), 1).flatten()[inds_yx].reshape(P, -1)
    )
    potential = 2 * torch.cumsum(weights_sorted, 1) - weights_sum
    out1 = torch.zeros_like(sorted_yx).flatten()
    out1[inds_yx] = potential.flatten()
    out1 = out1.reshape(P, -1)
    out1 = out1[:, :M]
    return out1


def fast_energy_summation_grad(x, y, x_weights, sliced_factor, batch_size, xis):
    # fast sum gradient via slicing and sorting
    d = x.shape[1]
    P = xis.shape[0]

    def with_projections(xi):
        P_local = xi.shape[0]
        x_proj = (xi @ x.T).reshape(P_local, -1)
        y_proj = (xi @ y.T).reshape(P_local, -1)
        outs = (
            -fastsum_energy_kernel_1D_grad(
                x_proj, x_weights[None, :].tile(P_local, 1), y_proj
            )
            * sliced_factor
        )
        return torch.nn.functional.conv1d(
            xi.squeeze().transpose(0, 1).flatten().reshape([1, 1, -1]),
            outs.transpose(0, 1).unsqueeze(1),
            stride=P_local,
        ).squeeze()

    return (
        torch.sum(
            torch.stack(
                [
                    with_projections(xis[i * batch_size : (i + 1) * batch_size])
                    for i in range((P - 1) // batch_size + 1)
                ],
                0,
            ),
            0,
        )
        / P
    )
