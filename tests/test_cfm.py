"""Tests for the conditional flow-matching implementation."""

import numpy as np
import pytest
import torch

from cfm import (
    SinusoidalTime,
    VelocityField,
    generate,
    interpolate,
    make_two_moons_labelled,
    mean_nn_distance,
    train,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_data_shapes_labels_and_standardisation():
    x, c = make_two_moons_labelled(1000, np.random.default_rng(0))
    assert x.shape == (1000, 2) and c.shape == (1000,)
    assert set(c.tolist()) == {0, 1}
    assert (c == 0).sum() == 500
    assert torch.allclose(x.mean(0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(x.std(0, unbiased=False), torch.ones(2), atol=1e-4)


def test_interpolate_endpoints():
    x0, x1 = torch.randn(5, 2), torch.randn(5, 2)
    assert torch.allclose(interpolate(x0, x1, torch.zeros(5)), x0)
    assert torch.allclose(interpolate(x0, x1, torch.ones(5)), x1)


def test_path_velocity_is_constant_and_equals_target():
    """d/dt x_t = x1 - x0 at every t: the regression target is exact."""
    x0, x1 = torch.randn(4, 2, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)
    h = 1e-6
    for t_val in (0.1, 0.5, 0.9):
        t = torch.full((4,), t_val, dtype=torch.float64)
        deriv = (interpolate(x0, x1, t + h) - interpolate(x0, x1, t - h)) / (2 * h)
        assert torch.allclose(deriv, x1 - x0, atol=1e-6)


class OracleVelocity(torch.nn.Module):
    """Exact marginal velocity when the data is a single point per class:
    v(x, t, c) = (target_c - x) / (1 - t). Lets us test the ODE sampler
    without any training."""

    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def forward(self, x, t, c):
        return (self.targets[c] - x) / (1 - t)[:, None]


def test_sampler_reaches_target_with_oracle_velocity():
    targets = torch.tensor([[1.5, -0.5], [-1.0, 2.0]])
    oracle = OracleVelocity(targets)
    for cls in (0, 1):
        out = generate(oracle, 200, cls=cls, n_steps=100)
        # With this field, explicit Euler lands exactly on the target at t = 1.
        assert torch.allclose(out, targets[cls].expand_as(out), atol=1e-4)


def test_generate_records_full_trajectory():
    model = VelocityField()
    x, traj = generate(model, 10, cls=0, n_steps=25, record=True)
    assert len(traj) == 26
    assert torch.equal(traj[-1], x)


def test_time_embedding_shape_and_distinct_times():
    emb = SinusoidalTime(64)(torch.tensor([0.0, 0.5, 1.0]))
    assert emb.shape == (3, 64)
    assert not torch.allclose(emb[0], emb[1])


def test_class_label_changes_the_prediction():
    model = VelocityField()
    x, t = torch.randn(8, 2), torch.rand(8)
    v0 = model(x, t, torch.zeros(8, dtype=torch.long))
    v1 = model(x, t, torch.ones(8, dtype=torch.long))
    assert v0.shape == (8, 2)
    assert not torch.allclose(v0, v1)


def test_training_reduces_loss():
    x, c = make_two_moons_labelled(1000, np.random.default_rng(0))
    model = VelocityField(hidden=64)
    losses = train(model, x, c, steps=300, batch_size=256, log_every=0)
    assert np.mean(losses[-50:]) < np.mean(losses[:50])


def test_mean_nn_distance():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert mean_nn_distance(a, a) == 0.0
    assert mean_nn_distance(a, np.array([[0.0, 1.0]])) == pytest.approx((1 + 2 ** 0.5) / 2)


def test_demo_smoke_run():
    import cfm_demo

    m = cfm_demo.main(["--steps", "30", "--no-figure"])
    for key in ("c0_to_moon0", "c1_to_moon1", "c0_correct_share", "real_to_real_floor"):
        assert key in m and np.isfinite(m[key])


def test_parameter_count_matches_readme():
    assert sum(p.numel() for p in VelocityField().parameters()) == 157_506
