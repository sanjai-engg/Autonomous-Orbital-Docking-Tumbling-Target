import numpy as np
import pytest
from spacecraft_docking import control


def test_sat_within_boundary():
    np.testing.assert_allclose(control.sat(np.array([0.5]), np.array([1.0])), [0.5])


def test_sat_outside_boundary():
    np.testing.assert_allclose(control.sat(np.array([2.0]), np.array([1.0])), [1.0])
    np.testing.assert_allclose(control.sat(np.array([-2.0]), np.array([1.0])), [-1.0])


def test_relative_quaternion_identity():
    q = np.array([1.0, 0, 0, 0])
    qr = control.relative_quaternion(q, q)
    np.testing.assert_allclose(qr, [1, 0, 0, 0], atol=1e-12)


def test_controller_constructs():
    cp = control.ControlParams.default()
    assert cp.lambda_p.shape == (3,)
    assert cp.k_a.shape == (3,)
    ctrl = control.SlidingModeController(
        chaser_mass=150.0, chaser_inertia=np.diag([26, 21, 31]),
        target_dock_offset=1.5, params=cp)
    assert ctrl.params.eps_p[0] > 0


def test_translational_control_finite():
    cp = control.ControlParams.default()
    ctrl = control.SlidingModeController(150.0, np.diag([26, 21, 31]), 1.5, cp)
    los = np.array([10.0, np.deg2rad(20), np.deg2rad(15)])
    los_dot = np.array([-0.5, np.deg2rad(-1), np.deg2rad(-1)])
    alpha_t, beta_t = np.deg2rad(15), np.deg2rad(10)
    u_los, s_p = ctrl.translational_control(los, los_dot, alpha_t, beta_t)
    assert u_los.shape == (3,)
    assert s_p.shape == (3,)
    assert np.all(np.isfinite(u_los))
    assert np.all(np.abs(u_los) <= cp.u_max + 1e-9)


def test_rotational_control_finite():
    cp = control.ControlParams.default()
    ctrl = control.SlidingModeController(150.0, np.diag([26, 21, 31]), 1.5, cp)
    q_c = np.array([1.0, 0, 0, 0])
    q_t = np.array([1.0, 0, 0, 0])
    omega_c = np.zeros(3)
    omega_t = np.array([0.01, 0.0, 0.0])
    tau, s_a = ctrl.rotational_control(q_c, omega_c, q_t, omega_t)
    assert tau.shape == (3,)
    assert s_a.shape == (3,)
    assert np.all(np.isfinite(tau))
    assert np.all(np.abs(tau) <= cp.tau_max + 1e-9)


def test_compute_returns_body_frame_u_and_tau():
    cp = control.ControlParams.default()
    ctrl = control.SlidingModeController(150.0, np.diag([26, 21, 31]), 1.5, cp)
    pos = np.array([3.0, 6.0, 7.0])
    vel = np.zeros(3)
    q_c = np.array([1.0, 0, 0, 0])
    omega_c = np.zeros(3)
    q_t = np.array([1.0, 0, 0, 0])
    omega_t = np.zeros(3)
    u_body, tau_body, info = ctrl.compute(0.0, pos, vel, q_c, omega_c, q_t, omega_t)
    assert u_body.shape == (3,)
    assert tau_body.shape == (3,)
    assert 's_p' in info and 's_a' in info
