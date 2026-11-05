"""Orbital disturbances acting on the chaser/target.

Paper §3.3 (translational disturbances) and §3.5 (rotational disturbances).
Aggregated into a single `Disturbances` object for plant injection.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Exponential atmosphere model (Vallado, around 400 km reference altitude)
RHO_REF = 3.614e-12      # reference density [kg/m³]
H_REF_M = 400e3          # reference altitude [m]
H_SCALE_M = 60.828e3     # scale height [m]


def atmospheric_drag(altitude_m: float, v_rel: np.ndarray,
                     C_D: float, area_m2: float, mass_kg: float) -> np.ndarray:
    """Aerodynamic drag acceleration on a spacecraft (paper Eq. 6).

    Density follows the exponential atmosphere model. Drag opposes velocity.
    """
    rho = RHO_REF * np.exp(-(altitude_m - H_REF_M) / H_SCALE_M)
    v_norm = np.linalg.norm(v_rel)
    if v_norm == 0.0:
        return np.zeros(3)
    return -0.5 * rho * (C_D * area_m2 / mass_kg) * v_norm * v_rel


def j2_perturbation(d_vec: np.ndarray, mu: float, R_E: float, J2: float) -> np.ndarray:
    """Acceleration from Earth's J2 oblateness for a body at d_vec (paper Eq. 8)."""
    d = np.linalg.norm(d_vec)
    factor = -1.5 * J2 * mu * R_E**2 / d**5
    z2_over_d2 = (d_vec[2] * d_vec[2]) / (d * d)
    return factor * np.array([
        (1 - 5 * z2_over_d2) * d_vec[0],
        (1 - 5 * z2_over_d2) * d_vec[1],
        (3 - 5 * z2_over_d2) * d_vec[2],
    ])


def gravity_gradient_torque(q: np.ndarray, I_body: np.ndarray,
                             mu: float, r_orbit: float) -> np.ndarray:
    """Gravity-gradient torque on body with attitude q (paper Eq. 21).

    Nadir vector (radial-in) is rotated into body frame and crossed with
    the inertia distribution. Vanishes when principal axes align with LVLH.
    """
    from .frames import quat_to_dcm
    R_lvlh_from_body = quat_to_dcm(q)
    nadir_lvlh = np.array([-r_orbit, 0.0, 0.0])           # radial-in
    d_body = R_lvlh_from_body.T @ nadir_lvlh              # nadir in body frame
    d_norm = np.linalg.norm(d_body)
    coeff = 3.0 * mu / d_norm**5
    Ixx, Iyy, Izz = I_body[0, 0], I_body[1, 1], I_body[2, 2]
    # T_GG = (3μ/r³) · n̂ × (I · n̂), expanded for diagonal I
    return coeff * np.array([
        (Izz - Iyy) * d_body[1] * d_body[2],
        (Ixx - Izz) * d_body[2] * d_body[0],
        (Iyy - Ixx) * d_body[0] * d_body[1],
    ])


@dataclass
class Disturbances:
    """Aggregator passed into Plant. Bundles drag + J2 + gravity-gradient inputs."""
    chaser_mass: float
    chaser_area: float
    chaser_C_D: float
    target_mass: float
    target_area: float
    target_C_D: float
    chaser_inertia: np.ndarray
    target_inertia: np.ndarray
    mu: float
    R_E: float
    J2: float
    orbit_altitude_m: float

    @property
    def orbit_radius_m(self) -> float:
        return self.R_E + self.orbit_altitude_m

    def translational(self, t: float, pos_rel: np.ndarray, vel_rel: np.ndarray) -> np.ndarray:
        """Disturbance acceleration on chaser in LVLH (paper §3.3).

        Drag applied to the chaser plus differential-J2 between chaser and target.
        """
        a_drag = atmospheric_drag(self.orbit_altitude_m, vel_rel,
                                  self.chaser_C_D, self.chaser_area, self.chaser_mass)
        d_chaser = np.array([self.orbit_radius_m + pos_rel[0], pos_rel[1], pos_rel[2]])
        d_target = np.array([self.orbit_radius_m, 0.0, 0.0])
        a_j2 = j2_perturbation(d_chaser, self.mu, self.R_E, self.J2) \
             - j2_perturbation(d_target, self.mu, self.R_E, self.J2)
        return a_drag + a_j2

    def gravity_gradient(self, q: np.ndarray, which: str) -> np.ndarray:
        """Gravity-gradient torque on chaser or target (paper §3.5, Eq. 21)."""
        I = self.chaser_inertia if which == 'chaser' else self.target_inertia
        return gravity_gradient_torque(q, I, self.mu, self.orbit_radius_m)
