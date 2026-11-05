"""Plant model: 20-dim relative dynamics (translation + chaser/target attitude).

Paper §3.2 (translational, Eq. 1) and §3.4 (rotational, Eqs 9, 14).
State vector layout: [pos(3), vel(3), q_c(4), ω_c(3), q_t(4), ω_t(3)].
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .frames import omega_skew, quat_to_dcm
from .disturbances import Disturbances


def pack_state(pos: np.ndarray, vel: np.ndarray,
               q_c: np.ndarray, omega_c: np.ndarray,
               q_t: np.ndarray, omega_t: np.ndarray) -> np.ndarray:
    """Concatenate the six pieces into a 20-vector for solve_ivp."""
    return np.concatenate([pos, vel, q_c, omega_c, q_t, omega_t])


def unpack_state(y: np.ndarray):
    """Inverse of pack_state — split the 20-vector into views."""
    return (y[0:3], y[3:6], y[6:10], y[10:13], y[13:17], y[17:20])


def orbital_rates(r_t: float, r_dot: float, mu: float) -> tuple[float, float]:
    """Mean motion n and ṅ for a target on an instantaneous radius (paper Eq. 2)."""
    n = np.sqrt(mu / r_t**3)
    ndot = -2.0 * (r_dot * n / r_t)
    return float(n), float(ndot)


def translational_acceleration(pos: np.ndarray, vel: np.ndarray,
                                n: float, ndot: float, mu: float, r_t: float,
                                thrust_accel: np.ndarray,
                                disturb_accel: np.ndarray) -> np.ndarray:
    """Relative translational acceleration in LVLH (paper Eq. 1).

    Returns (ẍ, ÿ, z̈). Includes Coriolis, centrifugal, gravity-asymmetry,
    thrust, and disturbance terms.
    """
    x, y, z = pos
    xdot, ydot, _zdot = vel
    r_c = np.sqrt((r_t + x)**2 + y**2 + z**2)
    ax = 2*n*ydot + n*n*x + ndot*y + mu/(r_t*r_t) - mu*(r_t + x)/r_c**3 \
         + thrust_accel[0] + disturb_accel[0]
    ay = -2*n*xdot + n*n*y - ndot*x - mu*y/r_c**3 \
         + thrust_accel[1] + disturb_accel[1]
    az = -mu*z/r_c**3 + thrust_accel[2] + disturb_accel[2]
    return np.array([ax, ay, az])


@dataclass
class Plant:
    """Closed-loop plant: 6-DOF translational + dual-attitude dynamics.

    Inputs to derivative(): chaser thrust acceleration in LVLH and chaser
    body torque. Target attitude is driven by gravity-gradient + an
    optional external torque (see scenarios for tumbling profiles).
    """
    chaser_mass: float
    chaser_inertia: np.ndarray
    target_inertia: np.ndarray
    disturbances: Disturbances
    target_external_torque: callable

    def derivative(self, t: float, y: np.ndarray,
                   u_chaser: np.ndarray, tau_chaser: np.ndarray) -> np.ndarray:
        """ẏ = f(t, y, u, τ) — full 20-vector time derivative for solve_ivp."""
        pos, vel, q_c, omega_c, q_t, omega_t = unpack_state(y)
        d = self.disturbances

        # Translational channel (paper Eq. 1)
        n, ndot = orbital_rates(d.orbit_radius_m, r_dot=0.0, mu=d.mu)
        f_dist = d.translational(t, pos, vel)
        accel = translational_acceleration(
            pos, vel, n=n, ndot=ndot, mu=d.mu, r_t=d.orbit_radius_m,
            thrust_accel=u_chaser, disturb_accel=f_dist)

        # Chaser rotational channel — Euler's equation + quaternion kinematics
        tau_grav_c = d.gravity_gradient(q_c, which='chaser')
        I_c = self.chaser_inertia
        omega_c_dot = np.linalg.solve(
            I_c, tau_chaser + tau_grav_c - np.cross(omega_c, I_c @ omega_c))
        q_c_dot = 0.5 * omega_skew(omega_c) @ q_c

        # Target rotational channel — gravity gradient + scenario disturbance
        tau_grav_t = d.gravity_gradient(q_t, which='target')
        tau_ext_t = self.target_external_torque(t)
        I_t = self.target_inertia
        omega_t_dot = np.linalg.solve(
            I_t, tau_grav_t + tau_ext_t - np.cross(omega_t, I_t @ omega_t))
        q_t_dot = 0.5 * omega_skew(omega_t) @ q_t

        return pack_state(vel, accel, q_c_dot, omega_c_dot, q_t_dot, omega_t_dot)
