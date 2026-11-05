"""Coordinate frames, quaternion algebra, and LVLH ↔ LOS transforms.

Paper: Abdollahzadeh & Esmailifar (2021), §3.1 (frames) and §3.4 (attitude).
Quaternion convention: scalar-first [q0, q1, q2, q3], unit norm.
"""
from __future__ import annotations
import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 (scalar-first). Used for attitude composition."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """q* — inverts rotation for a unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Renormalize to unit norm to fight integration drift."""
    return q / np.linalg.norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Body-to-LVLH DCM from a scalar-first quaternion (paper Eq. 17)."""
    q0, q1, q2, q3 = q
    return np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3),             2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),             q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),             2*(q2*q3 + q0*q1),             q0*q0 - q1*q1 - q2*q2 + q3*q3],
    ])


def omega_skew(omega: np.ndarray) -> np.ndarray:
    """Ω(ω) for quaternion kinematics q̇ = ½ Ω(ω) q (paper Eq. 12)."""
    wx, wy, wz = omega
    return np.array([
        [0,  -wx, -wy, -wz],
        [wx,  0,   wz, -wy],
        [wy, -wz,  0,   wx],
        [wz,  wy, -wx,  0],
    ])


def euler_to_quat(angles: np.ndarray) -> np.ndarray:
    """Z-Y-X intrinsic Euler (φ, θ, ψ) → scalar-first quaternion."""
    phi, theta, psi = angles
    cphi, sphi = np.cos(phi/2), np.sin(phi/2)
    cth,  sth  = np.cos(theta/2), np.sin(theta/2)
    cpsi, spsi = np.cos(psi/2), np.sin(psi/2)
    return np.array([
        cphi*cth*cpsi + sphi*sth*spsi,
        sphi*cth*cpsi - cphi*sth*spsi,
        cphi*sth*cpsi + sphi*cth*spsi,
        cphi*cth*spsi - sphi*sth*cpsi,
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion → Z-Y-X Euler angles (φ, θ, ψ) in radians."""
    q0, q1, q2, q3 = q
    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
    theta = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1, 1))
    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))
    return np.array([phi, theta, psi])


def lvlh_to_los(pos: np.ndarray) -> tuple[float, float, float]:
    """LVLH Cartesian (x,y,z) → LOS spherical (r, α elevation, β azimuth) per Eq. 3."""
    x, y, z = pos
    r = np.sqrt(x*x + y*y + z*z)
    beta = np.arctan2(-x, y)
    alpha = np.arctan2(z, np.sqrt(x*x + y*y))
    return float(r), float(alpha), float(beta)


def los_jacobian(alpha: float, beta: float) -> np.ndarray:
    """[T]^LS rotation matrix from LVLH to LOS (paper Eq. 4).

    Rows are the LOS-frame basis vectors expressed in LVLH:
      row 0 = e_r          (radial unit vector)
      row 1 = e_α (unit)   (direction of increasing elevation α)
      row 2 = e_β (unit)   (direction of increasing azimuth β)

    With α = arctan2(z, sqrt(x²+y²)) and β = arctan2(-x, y) per Eq. 3,
    so position = r·(-cos α sin β, cos α cos β, sin α).
    """
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    return np.array([
        [-sb*ca,   cb*ca,  sa],     # e_r
        [ sa*sb, -sa*cb,   ca],     # e_α
        [-cb,     -sb,     0 ],     # e_β
    ])


def docking_port_angles(q_target: np.ndarray, l_t: float) -> tuple[float, float]:
    """Target-port elevation α_t and azimuth β_t in LVLH from target attitude (Eq. 22).

    Port is along the target body +y axis at distance l_t from target COM.
    """
    R_lvlh_from_body = quat_to_dcm(q_target)
    port_lvlh = R_lvlh_from_body @ np.array([0.0, l_t, 0.0])
    lt_x, lt_y, lt_z = port_lvlh
    alpha_t = np.arctan2(lt_z, np.sqrt(lt_x*lt_x + lt_y*lt_y))
    beta_t  = np.arctan2(-lt_x, lt_y)
    return float(alpha_t), float(beta_t)


def docking_port_angle_rates(q_target: np.ndarray, omega_target_body: np.ndarray,
                              l_t: float) -> tuple[float, float]:
    """Time derivatives α̇_t, β̇_t for a tumbling target (feed-forward into SMC).

    Differentiates Eq. 22 using d(R_t·v_b)/dt = R_t·(ω_b × v_b).
    """
    R = quat_to_dcm(q_target)
    v_body = np.array([0.0, l_t, 0.0])
    port_lvlh = R @ v_body
    port_lvlh_dot = R @ np.cross(omega_target_body, v_body)
    lt_x, lt_y, lt_z = port_lvlh
    ltdot_x, ltdot_y, ltdot_z = port_lvlh_dot
    rho = np.sqrt(lt_x * lt_x + lt_y * lt_y)
    if rho < 1e-12 or l_t < 1e-12:
        return 0.0, 0.0
    rho_dot = (lt_x * ltdot_x + lt_y * ltdot_y) / rho
    alpha_t_dot = (ltdot_z * rho - lt_z * rho_dot) / (l_t * l_t)
    beta_t_dot  = (-ltdot_x * lt_y + lt_x * ltdot_y) / (rho * rho)
    return float(alpha_t_dot), float(beta_t_dot)
