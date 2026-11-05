"""Sensor model with Gaussian white noise per paper Table 4.

3σ specs from Table 4: position 1 cm, velocity 1 mm/s, attitude 0.25°,
angular rate 0.1°/s. σ values stored here are 3σ/3 to match.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import frames


@dataclass
class SensorModel:
    """Adds noise to the true state vector to feed the controller."""
    rng: np.random.Generator
    enabled: bool = True
    sigma_pos: float = 0.01 / 3.0                       # 3σ = 1 cm
    sigma_vel: float = 0.001 / 3.0                      # 3σ = 1 mm/s
    sigma_quat_rad: float = np.deg2rad(0.25 / 3.0)      # 3σ = 0.25°
    sigma_omega_rad: float = np.deg2rad(0.1 / 3.0)      # 3σ = 0.1°/s

    def measure(self, y: np.ndarray) -> np.ndarray:
        """Return a noisy measurement of the state. No-op when disabled."""
        if not self.enabled:
            return y.copy()
        out = y.copy()
        out[0:3]   += self.rng.normal(0.0, self.sigma_pos, size=3)
        out[3:6]   += self.rng.normal(0.0, self.sigma_vel, size=3)
        out[6:10]   = self._perturb_quat(out[6:10])
        out[10:13] += self.rng.normal(0.0, self.sigma_omega_rad, size=3)
        out[13:17]  = self._perturb_quat(out[13:17])
        out[17:20] += self.rng.normal(0.0, self.sigma_omega_rad, size=3)
        return out

    def _perturb_quat(self, q: np.ndarray) -> np.ndarray:
        """Compose a small-angle quaternion perturbation and renormalize."""
        delta_angle = self.rng.normal(0.0, self.sigma_quat_rad, size=3)
        dq = np.array([1.0, *(0.5 * delta_angle)])
        dq = frames.quat_normalize(dq)
        return frames.quat_normalize(frames.quat_multiply(dq, q))
