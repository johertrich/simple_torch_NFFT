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

try:
    import torchkbnufft as tkbn

    tkbn_comparison = True
except:
    print(
        "torchkbnufft cannot be loaded. Maybe it is not installed? Omit pyNFFT3 in the time comparison"
    )
    tkbn_comparison = False


device = "cuda" if torch.cuda.is_available() else "cpu"
double_precision = False
float_type = torch.float64 if double_precision else torch.float32
complex_type = torch.complex128 if double_precision else torch.complex64
m = 2
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


def rand_like(x):
    return (
        torch.rand_like(x)
        if isinstance(x, torch.Tensor)
        else np.random.uniform(size=x.shape)
    )


def randn_like(x):
    if isinstance(x, torch.Tensor):
        return torch.randn_like(x)
    elif x.dtype == np.complex128 or x.dtype == np.complex64:
        out = np.random.normal(size=x.shape) + 1j * np.random.normal(size=x.shape)
        return out.astype(x.dtype)
    elif x.dtype == np.float32 or x.dtype == np.float64:
        out = np.random.normal(size=x.shape)
        return out.astype(x.dtype)
    else:
        raise ValueError("unknown input type")


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
    outs = []
    for i in range(x.shape[0]):
        plan.x = x[i]
        outs_f = []
        for j in range(fhat.shape[1]):
            plan.fhat = fhat[i, j]
            plan.trafo()
            outs_f.append(plan.f)
        outs.append(np.stack(outs_f, 0))
    return np.stack(outs, 0)


def nfft3_adjoint(plan, x, f):
    outs = []
    for i in range(x.shape[0]):
        plan.x = x[0]
        outs_f = []
        for j in range(f.shape[1]):
            plan.f = f[i, j]
            plan.adjoint()
            outs_f.append(plan.fhat)
        outs.append(np.stack(outs_f, 0))
    return np.stack(outs, 0)


def torchkbnufft_forward(tkbn_obj, x, fhat):
    x_kb = 2 * torch.pi * x
    x_kb = x_kb.squeeze(1)
    x_kb = x_kb.transpose(-2, -1)
    return tkbn_obj(fhat, x_kb)


def torchkbnufft_adjoint(tkbn_adj, x, f):
    x_kb = 2 * torch.pi * x
    x_kb = x_kb.squeeze(1)
    x_kb = x_kb.transpose(-2, -1)
    return tkbn_adj(f, x_kb)


def run_test(method, runs, x, inp):
    toc_sum = 0
    for run in range(runs + 2):
        x = rand_like(x) - 0.5
        inp = randn_like(inp)
        sync()
        time.sleep(0.5)
        tic = time.time()
        res = method(x, inp)
        sync()
        toc = time.time() - tic
        # for some reasons torchkbnufft becomes faster after two runs.
        if run <= 1:
            continue
        toc_sum += toc
    return res, toc_sum


def test(N, M, batch_x, batch_f, runs=1):
    x = (
        torch.rand(
            (batch_x, 1, M, len(N)),
            device=device,
            dtype=float_type,
        )
        - 0.5
    )
    x_cpu = x.detach().cpu().numpy().astype(np.float64)
    x_cpu = np.ascontiguousarray(x_cpu.reshape(batch_x, M, len(N)))

    # init nfft
    nfft = NFFT(N, m=m, sigma=sigma, device=device, double_precision=double_precision)
    if nfft3_comparison:
        plan = pyNFFT3.NFFT(np.array(N, dtype="int32"), M, m=m)
    if tkbn_comparison:
        # we use window size=2*m, torchkbnufft window size = numpoints # table_oversampl is required that large to achieve
        # a small relative error...
        tkbn_obj = tkbn.KbNufft(im_size=N, numpoints=2 * m, table_oversamp=2**18).to(
            device
        )
        tkbn_adj = tkbn.KbNufftAdjoint(
            im_size=N, numpoints=2 * m, table_oversamp=2**18
        ).to(device)
        x_kb = 2 * torch.pi * x.clone()
        x_kb = x_kb.squeeze(1)
        x_kb = x_kb.transpose(-2, -1)

    f = torch.randn((batch_x, batch_f, M), dtype=complex_type, device=device)
    fHat_shape = [batch_x, batch_f] + list(N)
    fHat = torch.randn(fHat_shape, dtype=complex_type, device=device)
    # for pyNFFT3
    fHat_cpu = (
        fHat.detach()
        .cpu()
        .numpy()
        .astype(np.complex128)
        .squeeze()
        .reshape(batch_x, batch_f, -1)
    )
    f_cpu = (
        f.detach()
        .cpu()
        .numpy()
        .astype(np.complex128)
        .squeeze()
        .reshape(batch_x, batch_f, -1)
    )

    # compile
    out = nfft(x, fHat)
    out_adj = nfft.adjoint(x, f)
    batched_nfft(nfft, x, fHat.clone())
    batched_nfft(nfft.adjoint, x, f.clone())
    if tkbn_comparison:
        # we use window size=2*m, torchkbnufft window size = numpoints, therefore we set numpoints=2*m
        out_kb = tkbn_obj(fHat, x_kb)
        out_kb_adj = tkbn_adj(f, x_kb)

    # runtime forward
    _, toc = run_test(lambda x, fHat: nfft(x, fHat), runs, x, fHat)
    print("Simple forward:", toc)
    _, toc = run_test(lambda x, fHat: batched_nfft(nfft, x, fHat), runs, x, fHat)
    print("Batched forward:", toc)
    if tkbn_comparison:
        _, toc = run_test(
            lambda x, fHat: torchkbnufft_forward(tkbn_obj, x, fHat), runs, x, fHat
        )
        print("Torchkbnufft forward:", toc)
    if nfft3_comparison:
        _, toc = run_test(
            lambda x, fHat: nfft3_forward(plan, x, fHat), runs, x_cpu, fHat_cpu
        )
        print("NFFT3 forward:", toc)
    if torch_nfft_comparison:
        _, toc = run_test(lambda x, fHat: torch_nfft(x, fHat), runs, x, fHat)
        print("torch_nfft package forward:", toc)

    # runtime adjoint
    _, toc = run_test(lambda x, f: nfft.adjoint(x, f), runs, x, f)
    print("Simple adjoint:", toc)
    _, toc = run_test(lambda x, f: batched_nfft(nfft.adjoint, x, f), runs, x, f)
    print("Batched adjoint:", toc)
    if tkbn_comparison:
        _, toc = run_test(lambda x, f: torchkbnufft_adjoint(tkbn_adj, x, f), runs, x, f)
        print("Torchkbnufft adjoint:", toc)
    if nfft3_comparison:
        _, toc = run_test(lambda x, f: nfft3_adjoint(plan, x, f), runs, x_cpu, f_cpu)
        print("NFFT3 adjoint:", toc)
    if torch_nfft_comparison:
        _, toc = run_test(lambda x, f: torch_nfft_adjoint(x, f, N), runs, x, f)
        print("torch_nfft package forward:", toc)


M = 100000
runs = 10


print(f"all comparisons for M = {M} points of the non-equispaced grid")
for N in [(2**12,), (2**8, 2**8), (2**7, 2**7, 2**7)]:
    for batch_x in [1, 10]:
        for batch_f in [1, 10]:
            print(
                f"\n\nTest runtime for N={N}, batch_x={batch_x} and batch_f={batch_f}"
            )
            test(N, M, batch_x, batch_f, runs=runs)
