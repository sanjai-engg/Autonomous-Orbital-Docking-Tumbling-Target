import numpy as np
import pytest
from spacecraft_docking import frames


def test_quat_multiply_identity():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    e = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(frames.quat_multiply(q, e), q, atol=1e-12)
    np.testing.assert_allclose(frames.quat_multiply(e, q), q, atol=1e-12)


def test_quat_conjugate_inverse():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    qc = frames.quat_conjugate(q)
    prod = frames.quat_multiply(q, qc)
    np.testing.assert_allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_quat_normalize():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(frames.quat_normalize(q), [1.0, 0, 0, 0])


def test_quat_to_dcm_orthogonal():
    q = frames.quat_normalize(np.array([1.0, 0.5, -0.3, 0.2]))
    R = frames.quat_to_dcm(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_quat_to_dcm_identity():
    R = frames.quat_to_dcm(np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_dcm_rotates_vector():
    q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    R = frames.quat_to_dcm(q)
    np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)


def test_omega_skew_quaternion_kinematics():
    omega = np.array([0.0, 0.0, 1.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    qdot = 0.5 * frames.omega_skew(omega) @ q
    np.testing.assert_allclose(qdot, [0, 0, 0, 0.5], atol=1e-12)


def test_euler_to_quat_identity():
    q = frames.euler_to_quat(np.zeros(3))
    np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-12)


def test_euler_to_quat_roundtrip():
    angles = np.array([0.1, -0.2, 0.3])
    q = frames.euler_to_quat(angles)
    out = frames.quat_to_euler(q)
    np.testing.assert_allclose(out, angles, atol=1e-10)


def test_lvlh_to_los_known():
    pos = np.array([0.0, 5.0, 0.0])
    r, alpha, beta = frames.lvlh_to_los(pos)
    assert r == pytest.approx(5.0, abs=1e-12)
    assert alpha == pytest.approx(0.0, abs=1e-12)


def test_lvlh_to_los_roundtrip():
    pos = np.array([1.5, -3.2, 0.7])
    r, alpha, beta = frames.lvlh_to_los(pos)
    x = -r * np.cos(alpha) * np.sin(beta)
    y = r * np.cos(alpha) * np.cos(beta)
    z = r * np.sin(alpha)
    np.testing.assert_allclose([x, y, z], pos, atol=1e-10)


def test_los_jacobian_orthonormal_columns():
    T = frames.los_jacobian(0.5, 0.3)
    for col in range(3):
        assert np.linalg.norm(T[:, col]) == pytest.approx(1.0, abs=1e-12)


def test_los_jacobian_is_orthogonal():
    """Rows AND columns orthonormal so T.T @ u_LOS gives correct LVLH thrust vector."""
    T = frames.los_jacobian(0.4, -0.2)
    np.testing.assert_allclose(T @ T.T, np.eye(3), atol=1e-12)
    assert abs(abs(np.linalg.det(T)) - 1.0) < 1e-12


def test_los_jacobian_radial_velocity_sign():
    """Pure radial-outward LVLH velocity gives positive r_dot — guards against earlier sign bug."""
    pos = np.array([0.0, 5.0, 0.0])      # chaser on +y axis
    r, alpha, beta = frames.lvlh_to_los(pos)
    T = frames.los_jacobian(alpha, beta)
    v_radial_out = pos / r               # outward unit vector in LVLH
    los_vel = T @ v_radial_out
    assert los_vel[0] == pytest.approx(1.0, abs=1e-12)        # row 0 = e_r
    assert abs(los_vel[1]) < 1e-12 and abs(los_vel[2]) < 1e-12


def test_docking_port_angles_identity():
    q = np.array([1.0, 0, 0, 0])
    l_t = 1.5
    alpha_t, beta_t = frames.docking_port_angles(q, l_t)
    assert alpha_t == pytest.approx(0.0, abs=1e-12)
    assert beta_t == pytest.approx(0.0, abs=1e-12)


def test_docking_port_angle_rates_zero_when_static():
    q = np.array([1.0, 0, 0, 0])
    omega = np.zeros(3)
    a_dot, b_dot = frames.docking_port_angle_rates(q, omega, 1.5)
    assert a_dot == pytest.approx(0.0, abs=1e-12)
    assert b_dot == pytest.approx(0.0, abs=1e-12)


def test_docking_port_angle_rates_pitch_spin():
    """Target spinning about body x at 1 rad/s → port (along +y_b) sweeps in y-z plane.
    α_t increases at 1 rad/s, β_t = 0."""
    q = np.array([1.0, 0, 0, 0])
    omega = np.array([1.0, 0.0, 0.0])
    a_dot, b_dot = frames.docking_port_angle_rates(q, omega, 1.5)
    assert a_dot == pytest.approx(1.0, abs=1e-12)
    assert b_dot == pytest.approx(0.0, abs=1e-12)
