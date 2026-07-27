"""
Conditional flow matching from scratch (2-D)
============================================

A minimal, self-contained implementation of flow matching (Lipman et al.,
2023) trained with the conditional flow-matching objective, plus label
conditioning so we can steer *which* part of the distribution is generated.
Runs on CPU in well under a minute.

The task: the target is a two-moons distribution where each moon is a class
(0 = upper, 1 = lower). After training, asking for class 0 generates only the
upper moon and class 1 only the lower one -- so the model has learned a
*controllable* generator, not just an unconditional one.

How flow matching differs from diffusion
----------------------------------------
A diffusion model learns to *denoise* step by step. Flow matching instead
learns a **velocity field** that continuously transports a simple base
distribution (a Gaussian) into the data distribution along near-straight
paths, and generates by integrating an ODE. It is often simpler and needs
far fewer sampling steps.

The two ideas
-------------
1. Straight-line conditional paths. Pick a noise point x0 ~ N(0, I) and a
   real point x1. Interpolate linearly between them:

        x_t = (1 - t) * x0 + t * x1,      t in [0, 1]

   Along this straight path the ideal velocity is *constant*:

        u = x1 - x0

2. Regress the velocity. Train a network v_theta(x_t, t, c) -- also given the
   class label c -- to predict u. That single MSE objective is the whole of
   conditional flow matching. To generate a sample of class c, start from
   noise and integrate dx/dt = v_theta(x, t, c) from t = 0 to t = 1.

All data is synthetic. This is a mechanism demo, not a benchmark.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Target distribution, with a class label per point
# ---------------------------------------------------------------------------
# Two moons. Every point carries a label: 0 for the upper moon, 1 for the
# lower one. That label is what we will condition the generator on.
def make_two_moons_labelled(n, noise=0.06):
    n1 = n // 2
    n2 = n - n1
    t1 = np.pi * rng.uniform(0, 1, n1)
    t2 = np.pi * rng.uniform(0, 1, n2)
    moon1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)              # class 0
    moon2 = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)    # class 1
    x = np.concatenate([moon1, moon2], axis=0)
    c = np.concatenate([np.zeros(n1), np.ones(n2)]).astype(np.int64)
    x += rng.normal(0, noise, x.shape)
    mean, std = x.mean(0), x.std(0)
    x = (x - mean) / std                                           # standardise
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(c, dtype=torch.long))


data_x, data_c = make_two_moons_labelled(4000)


# ---------------------------------------------------------------------------
# 2. The velocity network v_theta(x, t, c)
# ---------------------------------------------------------------------------
# Inputs: the current point x, the continuous time t in [0, 1] (lifted with a
# sinusoidal embedding so the small MLP can use it smoothly), and the class
# label c (looked up in a small embedding table). Output: a 2-D velocity.
class SinusoidalTime(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        args = t[:, None] * freqs[None, :] * 100.0     # t in [0,1], scale up
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityField(nn.Module):
    def __init__(self, n_classes=2, t_dim=64, c_dim=32, hidden=256):
        super().__init__()
        self.time_embed = SinusoidalTime(t_dim)
        self.class_embed = nn.Embedding(n_classes, c_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim + c_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t, c):
        h = torch.cat([x, self.time_embed(t), self.class_embed(c)], dim=-1)
        return self.net(h)


model = VelocityField()


# ---------------------------------------------------------------------------
# 3. Training: the conditional flow-matching objective
# ---------------------------------------------------------------------------
# For each step: take real points (with their labels), pair each with a fresh
# noise point, place them on the straight path at a random time t, and train
# the network to output the constant path velocity (x1 - x0) for that label.
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
batch_size = 512
steps = 8000

for step in range(steps):
    idx = torch.randint(0, data_x.shape[0], (batch_size,))
    x1 = data_x[idx]                       # real point
    c = data_c[idx]                        # its class label
    x0 = torch.randn_like(x1)              # a noise point
    t = torch.rand(batch_size)             # random time in [0, 1]
    x_t = (1 - t)[:, None] * x0 + t[:, None] * x1     # point on the path
    target_velocity = x1 - x0              # constant along a straight path
    pred = model(x_t, t, c)
    loss = loss_fn(pred, target_velocity)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 2000 == 0:
        print(f"step {step + 1:>5d}/{steps}   loss {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 4. Sampling: integrate the ODE for a chosen class
# ---------------------------------------------------------------------------
# Start from Gaussian noise and step forward from t = 0 to t = 1 following the
# learned velocity, always conditioned on the requested class label. Flow
# matching needs only a handful of steps because the paths are near-straight;
# we use 100 here for smooth trajectories.
@torch.no_grad()
def generate(n, cls, n_steps=100, record=False):
    x = torch.randn(n, 2)
    c = torch.full((n,), cls, dtype=torch.long)
    dt = 1.0 / n_steps
    traj = [x.clone()]
    for i in range(n_steps):
        t = torch.full((n,), i * dt)
        x = x + model(x, t, c) * dt        # explicit Euler step
        if record:
            traj.append(x.clone())
    return (x, traj) if record else x


gen0 = generate(2000, cls=0)               # should be the upper moon
gen1 = generate(2000, cls=1)               # should be the lower moon
_, traj = generate(300, cls=0, record=True)  # trajectories for the figure


# ---------------------------------------------------------------------------
# 5. Quantitative check
# ---------------------------------------------------------------------------
# Confirm conditioning worked: class-0 samples should sit close to the real
# upper moon and far from the lower one, and vice versa.
real0 = data_x[data_c == 0].numpy()
real1 = data_x[data_c == 1].numpy()


def mean_nn(a, b):
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1).mean()


g0 = gen0.numpy(); g1 = gen1.numpy()
print("\nConditioning check (mean nearest-neighbour distance)")
print(f"  class-0 samples -> real upper moon : {mean_nn(g0, real0):.3f}   "
      f"(vs lower moon {mean_nn(g0, real1):.3f})")
print(f"  class-1 samples -> real lower moon : {mean_nn(g1, real1):.3f}   "
      f"(vs upper moon {mean_nn(g1, real0):.3f})")


# ---------------------------------------------------------------------------
# 6. Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))

ax[0].scatter(*real0.T, s=4, alpha=0.4, color="#1f4e79", label="class 0")
ax[0].scatter(*real1.T, s=4, alpha=0.4, color="#b5651d", label="class 1")
ax[0].set_title("Target distribution\n(two moons, one class each)")
ax[0].legend(fontsize=8, markerscale=2, loc="upper right")

ax[1].scatter(*g0.T, s=4, alpha=0.4, color="#1f4e79", label="asked for class 0")
ax[1].scatter(*g1.T, s=4, alpha=0.4, color="#b5651d", label="asked for class 1")
ax[1].set_title("Conditional generation\n(each class requested separately)")
ax[1].legend(fontsize=8, markerscale=2, loc="upper right")

# A few transport trajectories from noise (t=0) to data (t=1) for class 0.
traj_arr = torch.stack(traj).numpy()        # (n_steps+1, n_points, 2)
for k in range(0, traj_arr.shape[1], 8):
    ax[2].plot(traj_arr[:, k, 0], traj_arr[:, k, 1],
               color="#1f4e79", alpha=0.25, linewidth=0.7)
ax[2].scatter(*traj_arr[0].T, s=6, color="grey", alpha=0.6, label="start (noise)")
ax[2].scatter(*traj_arr[-1].T, s=6, color="#1f4e79", alpha=0.7, label="end (class 0)")
ax[2].set_title("Transport paths\n(noise flows to the requested class)")
ax[2].legend(fontsize=8, markerscale=2, loc="upper right")

for a in ax:
    a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5)
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")

fig.suptitle("Conditional flow matching: a learned velocity field transports "
             "noise into whichever moon (class) is requested.", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("figure.png", dpi=130)
print("\nSaved figure.png")
