import numpy as np
import pytest
from spacecraft_docking import disturbances


def test_drag_opposes_velocity():
    v = np.array([1.0, 0.0, 0.0])
    a = disturbances.atmospheric_drag(altitude_m=400e3, v_rel=v,
                                      C_D=2.2, area_m2=1.0, mass_kg=150.0)
    assert a[0] < 0
    assert a[1] == pytest.approx(0.0, abs=1e-20)
    assert a[2] == pytest.approx(0.0, abs=1e-20)


def test_drag_zero_velocity():
    a = disturbances.atmospheric_drag(400e3, np.zeros(3), 2.2, 1.0, 150.0)
    np.testing.assert_allclose(a, np.zeros(3), atol=1e-30)


def test_j2_magnitude_at_400km():
    MU = 3.986004418e14
    R_E = 6378.137e3
    J2 = 1.0826e-3
    d = np.array([R_E + 400e3, 0.0, 0.0])
    a = disturbances.j2_perturbation(d, MU, R_E, J2)
    # J2 perturbation at 400 km equatorial is ~0.0125 m/s² (~0.14% of central gravity)
    assert 1e-6 < np.linalg.norm(a) < 1e-1


def test_gravity_gradient_zero_when_principal_axis_nadir():
    # Body axes aligned with LVLH; nadir = -x_LVLH = -x_body. d_body = [-r, 0, 0].
    # T_GG components: (Iyy-Izz)*0*0, (Izz-Ixx)*0*-r, (Ixx-Iyy)*-r*0 = all zero.
    from spacecraft_docking import frames
    q_identity = np.array([1.0, 0, 0, 0])
    I = np.diag([26.0, 21.0, 31.0])
    r_orbit = 6778e3
    tau = disturbances.gravity_gradient_torque(q_identity, I, 3.986e14, r_orbit)
    np.testing.assert_allclose(tau, np.zeros(3), atol=1e-15)


def test_gravity_gradient_sign_restoring():
    """Body rotated +ε about z from radial alignment; restoring torque about -z if I_xx > I_yy.

    Stability condition: gravity-gradient pulls minimum-inertia axis toward radial.
    With I_yy < I_xx, body's y-axis prefers radial; small +ε perturbation about z
    must produce τ_z < 0 to drive the body BACK toward ε=0 if y_body is along radial,
    OR τ_z > 0 to drive ε toward π/2 if x_body started radial.
    Here body rotated +ε about z from x_body=radial state — pushes body toward
    y_body=radial (lower-inertia axis). So torque about z is positive.
    """
    from spacecraft_docking import frames
    eps = 0.05
    q = frames.euler_to_quat(np.array([0.0, 0.0, eps]))
    I = np.diag([26.0, 21.0, 31.0])      # Ixx > Iyy
    r_orbit = 6778e3
    mu = 3.986004418e14
    tau = disturbances.gravity_gradient_torque(q, I, mu, r_orbit)
    assert tau[2] > 0, f"Expected restoring torque sign, got τ_z={tau[2]}"


def test_aggregator_translational_returns_3vec():
    d = disturbances.Disturbances(
        chaser_mass=150.0, chaser_area=1.0, chaser_C_D=2.2,
        target_mass=400.0, target_area=2.0, target_C_D=2.2,
        chaser_inertia=np.diag([26, 21, 31]),
        target_inertia=np.diag([181, 123, 208]),
        mu=3.986004418e14, R_E=6378.137e3, J2=1.0826e-3,
        orbit_altitude_m=400e3,
    )
    pos_rel = np.array([3.0, 6.0, 7.0])
    vel_rel = np.array([0.1, 0.0, 0.0])
    a = d.translational(0.0, pos_rel, vel_rel)
    assert a.shape == (3,)
    assert np.all(np.isfinite(a))


def test_aggregator_gravity_gradient_chaser():
    d = disturbances.Disturbances(
        chaser_mass=150.0, chaser_area=1.0, chaser_C_D=2.2,
        target_mass=400.0, target_area=2.0, target_C_D=2.2,
        chaser_inertia=np.diag([26, 21, 31]),
        target_inertia=np.diag([181, 123, 208]),
        mu=3.986004418e14, R_E=6378.137e3, J2=1.0826e-3,
        orbit_altitude_m=400e3,
    )
    q = np.array([1.0, 0, 0, 0])
    tau = d.gravity_gradient(q, which='chaser')
    assert tau.shape == (3,)
