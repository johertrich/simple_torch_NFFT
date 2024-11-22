import torch
import numpy as np


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
