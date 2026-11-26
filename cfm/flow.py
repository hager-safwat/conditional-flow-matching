"""Conditional flow matching: probability path, training, ODE sampling, metric."""

import numpy as np
import torch
from torch import nn


def interpolate(x0, x1, t):
    """Straight-line conditional path x_t = (1 - t) * x0 + t * x1.

    x0, x1: (B, D). t: (B,). Its time derivative is the constant x1 - x0,
    which is the regression target of flow matching.
    """
    return (1 - t)[:, None] * x0 + t[:, None] * x1


def train(model, data_x, data_c, steps=8000, batch_size=512, lr=1e-3, log_every=2000):
    """Train v_theta with the conditional flow-matching loss.

    Each step: take real points with their labels, pair each with a fresh
    noise point, place them on the straight path at a random time, and
    regress the constant path velocity x1 - x0 (MSE).

    Note: the loss does not go to zero. Many (x0, x1) pairs pass through the
    same x_t, so the best possible prediction is their average velocity, and
    the leftover variance is irreducible.

    Returns the list of per-step losses.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for step in range(steps):
        idx = torch.randint(0, data_x.shape[0], (batch_size,))
        x1 = data_x[idx]
        c = data_c[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size)
        loss = loss_fn(model(interpolate(x0, x1, t), t, c), x1 - x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:>5d}/{steps}   loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def generate(model, n, cls, n_steps=100, data_dim=2, record=False):
    """Integrate dx/dt = v_theta(x, t, c) from t = 0 (noise) to t = 1 with
    explicit Euler steps, conditioned on class ``cls``.

    Returns the samples, plus the trajectory (list of tensors) if ``record``.
    """
    x = torch.randn(n, data_dim)
    c = torch.full((n,), cls, dtype=torch.long)
    dt = 1.0 / n_steps
    traj = [x.clone()]
    for i in range(n_steps):
        t = torch.full((n,), i * dt)
        x = x + model(x, t, c) * dt
        if record:
            traj.append(x.clone())
    return (x, traj) if record else x


def mean_nn_distance(a, b, chunk=1000):
    """Average distance from each point of ``a`` to its nearest point in ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mins = []
    for start in range(0, len(a), chunk):
        block = a[start:start + chunk]
        d = np.sqrt(((block[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        mins.append(d.min(axis=1))
    return float(np.concatenate(mins).mean())
