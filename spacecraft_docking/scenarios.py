"""Spacecraft physical parameters and scenario configurations.

Tables 1 (chaser/target masses + inertias), 2 (initial conditions), 3
(target tumbling profiles), and C2 (gain values, used in control.py).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


# Earth and orbital constants used throughout the simulation
MU = 3.986004418e14         # gravitational parameter [m³/s²]
R_E = 6378.137e3            # equatorial radius [m]
J2 = 1.0826e-3              # Earth oblateness coefficient
ORBIT_ALTITUDE_M = 400e3    # nominal target circular-orbit altitude (paper §5)


@dataclass(frozen=True)
class SpacecraftParams:
    """Physical params for one spacecraft (chaser or target). See Table 1."""
    mass: float                      # [kg]
    inertia: np.ndarray              # 3×3 body-frame inertia tensor [kg·m²]
    dimensions: np.ndarray           # box edge lengths [Lx, Ly, Lz] [m]
    docking_port_offset: float       # port distance along body +y_b [m]
    C_D: float = 2.2                 # drag coefficient

    @property
    def cross_section_m2(self) -> float:
        """Average projected drag area for an axis-tumbling cuboid."""
        Lx, Ly, Lz = self.dimensions
        return (Lx*Ly + Ly*Lz + Lx*Lz) / 3.0


CHASER = SpacecraftParams(
    mass=150.0, inertia=np.diag([26.0, 21.0, 31.0]),
    dimensions=np.array([1.2, 1.0, 0.8]), docking_port_offset=1.1)

TARGET = SpacecraftParams(
    mass=400.0, inertia=np.diag([181.0, 123.0, 208.0]),
    dimensions=np.array([2.0, 1.5, 1.2]), docking_port_offset=1.5)


@dataclass(frozen=True)
class Scenario:
    """Initial conditions + target torque profile for one simulation run (Tables 2, 3)."""
    name: str
    duration_s: float
    initial_pos_rel_m: np.ndarray
    initial_vel_rel_mps: np.ndarray
    initial_euler_chaser_deg: np.ndarray
    initial_euler_target_deg: np.ndarray
    initial_omega_chaser_body_dps: np.ndarray
    initial_omega_target_body_dps: np.ndarray
    external_torque_target: Callable[[float], np.ndarray]


def _zero_torque(t: float) -> np.ndarray:
    """Quiescent target — no external torque applied (scenarios 1, 2)."""
    return np.zeros(3)


def _sinusoidal_torque(t: float) -> np.ndarray:
    """Sinusoidal disturbance torque used to induce tumbling (scenarios 3, 4)."""
    return np.array([0.8, 0.5, -0.6]) * np.sin(0.12 * t)


_BASE_INITIALS = dict(
    initial_pos_rel_m=np.array([3.0, 6.0, 7.0]),
    initial_vel_rel_mps=np.zeros(3),
    initial_euler_chaser_deg=np.array([-10.0, 30.0, -15.0]),
    initial_euler_target_deg=np.array([20.0, 10.0, 15.0]),
    initial_omega_chaser_body_dps=np.zeros(3),
)


SCENARIOS: dict[int, Scenario] = {
    1: Scenario(
        name='no tumbling',
        duration_s=50.0,
        initial_omega_target_body_dps=np.zeros(3),
        external_torque_target=_zero_torque,
        **_BASE_INITIALS,
    ),
    2: Scenario(
        name='tumbling via initial angular velocity',
        duration_s=50.0,
        initial_omega_target_body_dps=np.array([1.5, 0.5, 0.0]),
        external_torque_target=_zero_torque,
        **_BASE_INITIALS,
    ),
    3: Scenario(
        name='tumbling via sinusoidal external torque',
        duration_s=70.0,
        initial_omega_target_body_dps=np.zeros(3),
        external_torque_target=_sinusoidal_torque,
        **_BASE_INITIALS,
    ),
    4: Scenario(
        name='tumbling via initial omega + sinusoidal torque',
        duration_s=70.0,
        initial_omega_target_body_dps=np.array([1.5, 0.5, 0.0]),
        external_torque_target=_sinusoidal_torque,
        **_BASE_INITIALS,
    ),
}
