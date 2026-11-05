import numpy as np
import pytest
from spacecraft_docking import dynamics


def test_pack_unpack_roundtrip():
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.1, 0.2, 0.3])
    qc = np.array([1.0, 0, 0, 0])
    wc = np.array([0.01, 0.02, 0.03])
    qt = np.array([0.7071, 0, 0, 0.7071])
    wt = np.array([0.1, 0.0, 0.0])
    y = dynamics.pack_state(pos, vel, qc, wc, qt, wt)
    assert y.shape == (20,)
    out = dynamics.unpack_state(y)
    np.testing.assert_allclose(out[0], pos)
    np.testing.assert_allclose(out[1], vel)
    np.testing.assert_allclose(out[2], qc)
    np.testing.assert_allclose(out[3], wc)
    np.testing.assert_allclose(out[4], qt)
    np.testing.assert_allclose(out[5], wt)


def test_orbital_rate_circular_400km():
    mu = 3.986004418e14
    r = 6378.137e3 + 400e3
    n, ndot = dynamics.orbital_rates(r, r_dot=0.0, mu=mu)
    assert n == pytest.approx(np.sqrt(mu / r**3), rel=1e-12)
    assert ndot == pytest.approx(0.0, abs=1e-20)


def test_translational_dynamics_zero_input_bounded():
    mu = 3.986004418e14
    r_t = 6378.137e3 + 400e3
    n = np.sqrt(mu / r_t**3)
    pos = np.array([1.0, 0.0, 0.0])
    vel = np.array([0.0, -2.0 * n * 1.0, 0.0])
    a = dynamics.translational_acceleration(
        pos, vel, n=n, ndot=0.0, mu=mu, r_t=r_t,
        thrust_accel=np.zeros(3), disturb_accel=np.zeros(3))
    assert np.all(np.isfinite(a))
    assert np.linalg.norm(a) < 0.1


def test_plant_derivative_shape():
    from spacecraft_docking.disturbances import Disturbances
    dist = Disturbances(150, 1.0, 2.2, 400, 2.0, 2.2,
                        np.diag([26, 21, 31]), np.diag([181, 123, 208]),
                        3.986004418e14, 6378.137e3, 1.0826e-3, 400e3)
    plant = dynamics.Plant(
        chaser_mass=150.0, chaser_inertia=np.diag([26, 21, 31]),
        target_inertia=np.diag([181, 123, 208]),
        disturbances=dist, target_external_torque=lambda t: np.zeros(3))
    y0 = dynamics.pack_state(
        pos=np.array([3, 6, 7], float), vel=np.zeros(3),
        q_c=np.array([1, 0, 0, 0], float), omega_c=np.zeros(3),
        q_t=np.array([1, 0, 0, 0], float), omega_t=np.zeros(3))
    dy = plant.derivative(0.0, y0, np.zeros(3), np.zeros(3))
    assert dy.shape == (20,)
    assert np.all(np.isfinite(dy))
