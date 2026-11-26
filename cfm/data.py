"""Target distribution: two half-moons in 2-D, one class label per moon."""

import numpy as np
import torch


def make_two_moons_labelled(n, rng, noise=0.06):
    """Sample n labelled points: class 0 = upper moon, class 1 = lower moon.

    Parameters
    ----------
    n : int
        Number of points (split evenly between the two classes).
    rng : np.random.Generator
        Source of randomness, passed in explicitly for reproducibility.
    noise : float
        Standard deviation of the Gaussian jitter added before standardising.

    Returns
    -------
    x : torch.Tensor, shape (n, 2), float32, standardised
    c : torch.Tensor, shape (n,), int64 class labels
    """
    n1 = n // 2
    n2 = n - n1
    t1 = np.pi * rng.uniform(0, 1, n1)
    t2 = np.pi * rng.uniform(0, 1, n2)
    moon1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)              # class 0
    moon2 = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)    # class 1
    x = np.concatenate([moon1, moon2], axis=0)
    c = np.concatenate([np.zeros(n1), np.ones(n2)]).astype(np.int64)
    x += rng.normal(0, noise, x.shape)
    x = (x - x.mean(0)) / x.std(0)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(c, dtype=torch.long)
