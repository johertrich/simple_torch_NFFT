from simple_torch_NFFT import NFFT
import torch
import time
import numpy as np

try:
    import torch_nfft as tn

    if torch.cuda.is_available():
        torch_nfft_comparison = True
    else:
        print("cuda not available. Omit torch_nfft in the time comparison")
        torch_nfft_comparison = False
except:
    print("torch_nfft cannot be loaded. Omit torch_nfft in the time comparison")
    torch_nfft_comparison = False

try:
    import pyNFFT3

    nfft3_comparison = True
except:
    print(
        "pyNFFT3 cannot be loaded. Maybe it is not installed? Omit pyNFFT3 in the time comparison"
    )
    nfft3_comparison = False

device = "cuda" if torch.cuda.is_available() else "cpu"
double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64
m = 4
sigma = 2

# for gpu testing
sync = (
    (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)
)


def batched_nfft(nfft_fun, points, inp):
    torch.cat(
        [
            torch.cat(
                [
                    nfft_fun(points[i : i + 1], inp[i : i + 1, j : j + 1])
                    for j in range(inp.shape[1])
                ],
                1,
            )
            for i in range(points.shape[0])
        ],
        0,
    )


def torch_nfft(x, fHat):
    x = x.tile(1, fHat.shape[1], 1, 1)
    batch = torch.arange(x.shape[0] * x.shape[1], device=device).repeat_interleave(
        x.shape[2]
    )
    fHat_shape = [-1] + list(fHat.shape[2:])
    return tn.nfft_forward(
        fHat.view(fHat_shape),
        x.view(-1, x.shape[-1]),
        batch=batch,
        cutoff=m,
    ).view(x.shape[0], fHat.shape[1], -1)


def torch_nfft_adjoint(x, f, N):
    x = x.tile(1, f.shape[1], 1, 1)
    batch = torch.arange(x.shape[0] * x.shape[1], device=device).repeat_interleave(
        x.shape[2]
    )
    out_shape = [x.shape[0], f.shape[1]] + list(N)
    return tn.nfft_adjoint(
        f.flatten(), x.view(-1, x.shape[-1]), bandwidth=N[0], batch=batch, cutoff=m
    ).view(out_shape)


def nfft3_forward(plan, x, fhat):
    plan.fhat = fhat
    plan.x = x
    plan.trafo()
    return plan.f


def run_test(method, runs):
    sync()
    tic = time.time()
    for _ in range(runs):
        res = method()
        sync()
    toc = time.time() - tic
    return res, toc


def test(N, J, batch_x, batch_f, runs=1):
    x = (
        torch.rand(
            (batch_x, 1, J, len(N)),
            device=device,
            dtype=float_type,
        )
        - 0.5
    )
    x_cpu = x.detach().cpu().numpy().astype(np.float64)
    x_cpu = np.ascontiguousarray(x_cpu.squeeze())

    # init nfft
    nfft = NFFT(N, m=m, sigma=sigma, device=device, double_precision=double_precision)
    if nfft3_comparison:
        plan = pyNFFT3.NFFT(np.array(N, dtype="int32"), J)

    f = torch.randn((batch_x, batch_f, J), dtype=complex_type, device=device)
    fHat_shape = [batch_x, batch_f] + list(N)
    fHat = torch.randn(fHat_shape, dtype=complex_type, device=device)
    # for pyNFFT3
    fHat_cpu = fHat.detach().cpu().numpy().astype(np.complex128).squeeze().reshape(-1)

    # compile
    nfft(x, fHat)
    nfft.adjoint(x, f)
    batched_nfft(nfft, x, fHat)
    batched_nfft(nfft.adjoint, x, f)

    # runtime forward
    _, toc = run_test(lambda: nfft(x, fHat), runs)
    print("Simple forward:", toc)
    _, toc = run_test(lambda: batched_nfft(nfft, x, fHat), runs)
    print("Batched forward:", toc)
    _, toc = run_test(lambda: nfft3_forward(plan, x_cpu, fHat_cpu), runs)
    print("NFFT3 forward:", toc)
    if torch_nfft_comparison:
        _, toc = run_test(lambda: torch_nfft(x, fHat), runs)
        print("torch_nfft package forward:", toc)

    # runtime adjoint
    _, toc = run_test(lambda: nfft.adjoint(x, f), runs)
    print("Simple adjoint:", toc)
    _, toc = run_test(lambda: batched_nfft(nfft.adjoint, x, f), runs)
    print("Batched adjoint:", toc)
    if torch_nfft_comparison:
        _, toc = run_test(lambda: torch_nfft_adjoint(x, f, N), runs)
        print("torch_nfft package forward:", toc)


N = (2**5, 2**5, 2**5)
batch_x = 1
batch_f = 1
J = 100000
runs = 1

test(N, J, batch_x, batch_f)