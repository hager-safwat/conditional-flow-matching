"""
Conditional flow matching from scratch (2-D)
============================================

Trains a label-conditioned velocity field with the conditional flow-matching
objective (Lipman et al., 2023) on a two-moons distribution where each moon
is a class, generates each class separately by integrating the learned ODE,
checks that conditioning really controls which moon comes out, and draws the
figure.

All data is synthetic. This is a mechanism demo, not a benchmark.

Usage
-----
    python cfm_demo.py                          # full run, writes figure.png
    python cfm_demo.py --steps 300 --no-figure  # quick smoke run
"""

import argparse

import numpy as np
import torch

from cfm import VelocityField, generate, make_two_moons_labelled, mean_nn_distance, train

SEED = 0
N_POINTS = 4000


def nearest_class(points, real0, real1, chunk=1000):
    """For each point, return 0 if its nearest real point is in class 0, else 1."""
    out = []
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        d0 = np.sqrt(((p[:, None] - real0[None]) ** 2).sum(-1)).min(1)
        d1 = np.sqrt(((p[:, None] - real1[None]) ** 2).sum(-1)).min(1)
        out.append((d1 < d0).astype(int))
    return np.concatenate(out)


def run_experiment(steps=8000, seed=SEED, log_every=2000):
    """Train, generate and evaluate. Returns (metrics, arrays for the figure).

    Order matters for exact reproducibility: one numpy generator makes the
    data, one torch seed controls initialisation, training and sampling.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    data_x, data_c = make_two_moons_labelled(N_POINTS, rng)
    model = VelocityField()
    losses = train(model, data_x, data_c, steps=steps, log_every=log_every)

    gen0 = generate(model, 2000, cls=0).numpy()
    gen1 = generate(model, 2000, cls=1).numpy()
    _, traj = generate(model, 300, cls=0, record=True)

    real0 = data_x[data_c == 0].numpy()
    real1 = data_x[data_c == 1].numpy()

    # Reference floor: a fresh draw of REAL data from a separate generator.
    fresh_x, fresh_c = make_two_moons_labelled(N_POINTS, np.random.default_rng(seed + 1))
    fresh0 = fresh_x[fresh_c == 0].numpy()

    metrics = {
        "loss_last_1000_mean": float(np.mean(losses[-1000:])),
        "c0_to_moon0": mean_nn_distance(gen0, real0),
        "c0_to_moon1": mean_nn_distance(gen0, real1),
        "c1_to_moon1": mean_nn_distance(gen1, real1),
        "c1_to_moon0": mean_nn_distance(gen1, real0),
        "real_to_real_floor": mean_nn_distance(fresh0, real0),
        "c0_correct_share": float((nearest_class(gen0, real0, real1) == 0).mean()),
        "c1_correct_share": float((nearest_class(gen1, real0, real1) == 1).mean()),
    }
    arrays = {"real0": real0, "real1": real1, "gen0": gen0, "gen1": gen1,
              "traj": torch.stack(traj).numpy()}
    return metrics, arrays


def print_metrics(m):
    print(f"\ntraining loss (mean of last 1000)          {m['loss_last_1000_mean']:.4f}")
    print("\nConditioning check (mean nearest-neighbour distance, standardised units)")
    print(f"  class-0 samples -> upper moon {m['c0_to_moon0']:.3f}   "
          f"(vs lower moon {m['c0_to_moon1']:.3f})")
    print(f"  class-1 samples -> lower moon {m['c1_to_moon1']:.3f}   "
          f"(vs upper moon {m['c1_to_moon0']:.3f})")
    print(f"  real -> real, fresh draw (floor)  {m['real_to_real_floor']:.3f}")
    print("\nShare of samples whose nearest real point has the requested class")
    print(f"  class 0: {100 * m['c0_correct_share']:.1f} %   "
          f"class 1: {100 * m['c1_correct_share']:.1f} %")


def save_figure(a, path="figure.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    ax[0].scatter(*a["real0"].T, s=4, alpha=0.4, color="#1f4e79", label="class 0")
    ax[0].scatter(*a["real1"].T, s=4, alpha=0.4, color="#b5651d", label="class 1")
    ax[0].set_title("Target distribution\n(two moons, one class each)")
    ax[0].legend(fontsize=8, markerscale=2, loc="upper right")

    ax[1].scatter(*a["gen0"].T, s=4, alpha=0.4, color="#1f4e79", label="asked for class 0")
    ax[1].scatter(*a["gen1"].T, s=4, alpha=0.4, color="#b5651d", label="asked for class 1")
    ax[1].set_title("Conditional generation\n(each class requested separately)")
    ax[1].legend(fontsize=8, markerscale=2, loc="upper right")

    traj = a["traj"]                                    # (n_steps + 1, n_points, 2)
    for k in range(0, traj.shape[1], 8):
        ax[2].plot(traj[:, k, 0], traj[:, k, 1], color="#1f4e79", alpha=0.25, linewidth=0.7)
    ax[2].scatter(*traj[0].T, s=6, color="grey", alpha=0.6, label="start (noise)")
    ax[2].scatter(*traj[-1].T, s=6, color="#1f4e79", alpha=0.7, label="end (class 0)")
    ax[2].set_title("Transport paths\n(noise flows to the requested class)")
    ax[2].legend(fontsize=8, markerscale=2, loc="upper right")

    for axis in ax:
        axis.set_xlim(-2.5, 2.5)
        axis.set_ylim(-2.5, 2.5)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")

    fig.suptitle("Conditional flow matching: a learned velocity field transports "
                 "noise into whichever moon (class) is requested.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=130)
    print(f"\nSaved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a 2-D conditional flow-matching model.")
    parser.add_argument("--steps", type=int, default=8000, help="training steps")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-figure", action="store_true", help="skip figure.png")
    args = parser.parse_args(argv)

    metrics, arrays = run_experiment(steps=args.steps, seed=args.seed,
                                     log_every=max(1, args.steps // 4))
    print_metrics(metrics)
    if not args.no_figure:
        save_figure(arrays)
    return metrics


if __name__ == "__main__":
    main()
