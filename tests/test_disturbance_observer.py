"""Tests for the RLS+RFF disturbance observer.

Two layers of testing:
  1. RLSObserver as a black-box learner — verify it can fit a known nonlinear
     map online and that its residual decreases with iterations.
  2. DisturbanceObserver as a paired translational/rotational wrapper — verify
     interface and that disabled observers return zero predictions.
"""
import numpy as np
import pytest

from spacecraft_docking.disturbance_observer import RLSObserver, DisturbanceObserver


def test_rls_fits_constant_offset():
    """Constant disturbance: observer should converge to it after a few steps."""
    rng = np.random.default_rng(0)
    obs = RLSObserver(input_dim=4, output_dim=3, n_features=32,
                       sigma=2.0, forgetting=1.0, reg=1e-2, rng=rng)
    target = np.array([0.5, -0.2, 0.1])
    x = np.zeros(4)
    for _ in range(50):
        obs.update(x, target)
    pred = obs.predict(x)
    np.testing.assert_allclose(pred, target, atol=1e-3)


def test_rls_fits_nonlinear_map():
    """Quadratic map y(x) = (x1², x1·x2, x2²) — observer must reduce residual."""
    rng = np.random.default_rng(1)
    obs = RLSObserver(input_dim=2, output_dim=3, n_features=128,
                       sigma=1.0, forgetting=1.0, reg=1e-2, rng=rng)

    def truth(x):
        return np.array([x[0]**2, x[0] * x[1], x[1]**2])

    # Train on a small grid
    grid = np.linspace(-1, 1, 11)
    for _ in range(3):  # multiple passes
        for a in grid:
            for b in grid:
                x = np.array([a, b])
                obs.update(x, truth(x))

    # Test set: same range, finer grid
    test_pts = rng.uniform(-1, 1, size=(50, 2))
    err = np.array([np.linalg.norm(obs.predict(x) - truth(x)) for x in test_pts])
    assert err.mean() < 0.05, f"Mean error too large: {err.mean()}"


def test_rls_residual_decreases():
    """During fitting, residual norm trends downward."""
    rng = np.random.default_rng(2)
    obs = RLSObserver(input_dim=3, output_dim=2, n_features=64,
                       sigma=1.5, forgetting=1.0, reg=1e-2, rng=rng)
    target_w = rng.normal(size=(2, 3))    # linear ground truth
    early_residuals, late_residuals = [], []
    for k in range(200):
        x = rng.normal(size=3)
        y = target_w @ x
        res = obs.update(x, y)
        if k < 20:
            early_residuals.append(np.linalg.norm(res))
        elif k > 150:
            late_residuals.append(np.linalg.norm(res))
    assert np.mean(late_residuals) < 0.5 * np.mean(early_residuals)


def test_disturbance_observer_disabled_returns_zero():
    obs = DisturbanceObserver(enabled=False)
    z3 = np.zeros(3)
    z6 = np.zeros(6)[:3]   # dummies
    np.testing.assert_array_equal(obs.predict_translational(z3, z3), np.zeros(3))
    np.testing.assert_array_equal(obs.predict_rotational(z3, z3), np.zeros(3))


def test_disturbance_observer_paired_interfaces():
    obs = DisturbanceObserver()
    e_p     = np.array([1.0, 0.05, -0.04])
    e_p_dot = np.array([-0.1, 0.0,  0.0])
    target_d = np.array([0.001, -0.002, 0.0005])
    for _ in range(100):
        obs.update_translational(e_p, e_p_dot, target_d)
    pred = obs.predict_translational(e_p, e_p_dot)
    np.testing.assert_allclose(pred, target_d, atol=5e-4)


def test_observer_does_not_explode_on_zero_input():
    obs = DisturbanceObserver()
    pred = obs.predict_translational(np.zeros(3), np.zeros(3))
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))
