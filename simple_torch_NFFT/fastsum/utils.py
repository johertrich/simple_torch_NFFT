import torch
import numpy as np
import scipy
import mpmath


def get_median_distance(x, y, batch_size=1000):
    perm1 = torch.randperm(x.shape[0])[:batch_size]
    perm2 = torch.randperm(y.shape[0])[:batch_size]
    dists = torch.sqrt(
        torch.sum((x[perm1][:, None, :] - y[perm2][None, :, :]) ** 2, -1)
    )
    return torch.median(dists.reshape(-1))


def compute_sliced_factor(d):
    # Compute the slicing constant within the negative distance kernel
    k = (d - 1) // 2
    fac = 1.0
    if (d - 1) % 2 == 0:
        for j in range(1, k + 1):
            fac = 2 * fac * j / (2 * j - 1)
    else:
        for j in range(1, k + 1):
            fac = fac * (2 * j + 1) / (2 * j)
        fac = fac * np.pi / 2
    return fac


def compute_thin_plate_constant(d):
    if d % 2 == 0:
        mysum = 0.0
        for k in range(1, d // 2 + 1):
            mysum += 1 / k
    else:
        # naive numerical integration for computing the harmonic number
        # for non-integral d/2...
        pieces = 1000
        grid = np.linspace(0, 1 - 1 / pieces, pieces)
        integrands = (1 - grid ** (d / 2)) / (1 - grid)
        mysum = np.sum(integrands) / pieces
    return -d / 2 * (mysum - 2 + np.log(4.0))


def compute_logarithmic_constant(d):
    mpmath.mp.dps = 25
    mpmath.mp.pretty = True
    other_factor = (
        2
        * np.exp(scipy.special.loggamma(d / 2) - scipy.special.loggamma((d - 1) / 2))
        / np.sqrt(np.pi)
    )
    return other_factor * float(mpmath.hyp3f2(0.5, 0.5, -0.5 * (d - 3), 1.5, 1.5, 1))
