import torch
from simple_torch_NFFT import NFFT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters of the NFFT
N = (2**10,)  # size of the regular grid as tuple, here (in 1D) 1024.

# create NFFT object
nfft = NFFT(N)
# optional arguments are
# m (window size, default 4)
# n (size of the oversampled grid (tuple), None for using oversampling factor, default: None)
# sigma (oversampling factor, default 2, not used if n is given)
# window (default simple_torch_NFFT.KaiserBesselWindow, other option simple_torch_NFFT.GaussWindow)
# device (default "cuda" if torch.cuda.is_available() else "cpu")
# double_precision (default false)
# no_compile (set to True to supress compile, mainly useful for debugging to get readible stack traces, default: False)

# Parameters of the input
M = 20000  # number of basis points
batch_x = 2  # batches of basis points
batch_f = 2  # batches of function values
# basis points, NFFT will be taken wrt the second dimension
x = (torch.rand((batch_x, 1, M, len(N),), device=device,) - 0.5 )

# forward NFFT
f_hat_shape = [batch_x, batch_f] + list(N)  # f_hat has batch dimensions + grid dimensions
f_hat = torch.randn(f_hat_shape, dtype=torch.complex64, device=device)  # Fourier coefficients
f = nfft(x, f_hat)

# adjoint NFFT
f = torch.randn((batch_x, batch_f, M), dtype=torch.complex64, device=device)  # function values
f_hat = nfft.adjoint(x, f)
