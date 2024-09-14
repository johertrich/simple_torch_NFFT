import torch
from simple_torch_NFFT import NFFT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters of the NFFT
N = 2**10  # number of Fourier coefficients

# create NFFT object
nfft = NFFT(N)
# optional arguments are
# m (window size, default 4)
# n (oversampled number of Fourier coefficients, None for using oversampling factor, default: None)
# sigma (oversampling factor, default 2, not used if n is given)
# window (default simple_torch_NFFT.KaiserBesselWindow, other option simple_torch_NFFT.GaussWindow)
# device (default "cuda" if torch.cuda.is_available() else "cpu")
# double_precision (default false)

# Parameters of the input
M = 20000  # number of basis points
batch_x = 2  # batches of basis points
batch_f = 2  # batches of function values
# basis points, NFFT will be taken wrt the second dimension
k = (torch.rand((batch_x, 1, M,), device=device,) - 0.5 )

# forward NFFT
f_hat = torch.randn(
    (k.shape[0], batch_f, N), dtype=torch.complex64, device=device
)  # Fourier coefficients
f = nfft(k, f_hat)

# adjoint NFFT
f = torch.randn(k.shape, dtype=torch.complex64, device=device)  # function values
f_hat = nfft.adjoint(k, f)
