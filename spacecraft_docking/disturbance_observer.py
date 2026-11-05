"""Online disturbance observer via recursive least squares with random
Fourier features.

The observer learns a vector-valued residual map  x ↦ d̂(x)  online during
flight. The hypothesis class is linear in random Fourier features
phi(x) = sqrt(2/D) cos(W x + b), which approximates an RBF-kernel regression
(Rahimi & Recht 2007) at fixed feature width and is therefore close in spirit
to kernel SVR while remaining O(D²) per step.

The observer is updated once per ZOH control step with the residual between
measured plant acceleration / angular acceleration and the nominal model
prediction. The current prediction is fed forward into the SMC equivalent
control, which lets the reaching-law gain be lowered without sacrificing
robustness — directly reducing chatter on the commanded thrust and torque.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RLSObserver:
    """Recursive least squares with random Fourier features.

    Parameters
    ----------
    input_dim   : feature-vector length
    output_dim  : disturbance dimensionality (3 for accel or torque)
    n_features  : RFF width D (more = more capacity, slower)
    sigma       : RBF kernel bandwidth (larger = smoother fit)
    forgetting  : RLS forgetting factor λ ∈ (0, 1]; smaller adapts faster
    reg         : initial covariance regularisation; smaller = more conservative
    rng         : numpy Generator for reproducibility
    """
    input_dim: int
    output_dim: int
    n_features: int = 64
    sigma: float = 1.0
    forgetting: float = 0.995
    reg: float = 1e-2
    rng: np.random.Generator | None = None

    # learnable / state
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    scale: float = field(init=False)
    theta: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # The random Fourier feature map is fixed at construction time.
        # W is sampled from N(0, 1/sigma²) — a draw from the spectral measure of
        # the Gaussian RBF kernel, per Bochner's theorem. b is uniform on [0, 2π].
        # phi(x) = sqrt(2/D) cos(W x + b) then satisfies E[phi(x).phi(y)] ≈ k(x,y)
        # for the Gaussian RBF kernel with bandwidth `sigma`.
        rng = self.rng if self.rng is not None else np.random.default_rng(0)
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(self.n_features, self.input_dim))
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)
        self.scale = np.sqrt(2.0 / self.n_features)        # variance correction
        # theta: D-by-output_dim weight matrix; one row per output dimension.
        self.theta = np.zeros((self.output_dim, self.n_features))
        # P: scaled inverse covariance of theta. Large initial P (small reg)
        # = uninformative prior — early observations move theta aggressively.
        self.P = np.eye(self.n_features) / self.reg

    # ------------------------------------------------------------------ ops
    def features(self, x: np.ndarray) -> np.ndarray:
        """phi(x) — random Fourier feature embedding of input x."""
        return self.scale * np.cos(self.W @ x + self.b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Estimated disturbance d̂(x) ∈ R^{output_dim}."""
        return self.theta @ self.features(x)

    def update(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Standard RLS update with forgetting on a sample (x, y).

        Algorithm (per Haykin, Adaptive Filter Theory, Ch. 13):
          phi   = phi(x)                          # feature vector, R^D
          Pphi  = P phi                           # covariance × feature, R^D
          denom = lambda + phi^T P phi            # scalar
          k     = Pphi / denom                    # Kalman gain, R^D
          theta ← theta + (y - theta phi) k^T     # one update per output dim
          P     ← (P - k phi^T P) / lambda        # downdate covariance

        The forgetting factor lambda < 1 gives an exponential window of
        effective length 1/(1 - lambda) past samples, after which old data is
        gracefully forgotten. This is what lets the observer track slowly
        time-varying disturbances rather than averaging them away.

        Returns the *pre-update* prediction residual y − d̂(x) so callers can
        log convergence diagnostics without computing it twice.
        """
        phi  = self.features(x)
        Pphi = self.P @ phi
        denom = self.forgetting + float(phi @ Pphi)
        gain = Pphi / denom
        residual = y - self.theta @ phi
        # Rank-one update: theta gains residual·gain^T (one row per output).
        self.theta = self.theta + np.outer(residual, gain)
        # Sherman-Morrison covariance downdate, scaled by 1/lambda for forgetting.
        self.P = (self.P - np.outer(gain, Pphi)) / self.forgetting
        return residual

    def reset_covariance(self, reg: float | None = None) -> None:
        """Re-initialise covariance, e.g. on regime change."""
        r = reg if reg is not None else self.reg
        self.P = np.eye(self.n_features) / r


@dataclass
class DisturbanceObserver:
    """Pair of RLS observers — one for translational, one for rotational.

    Two channels are kept independent so each can use its own kernel bandwidth
    and forgetting factor; the translational disturbance (drag, J2 differential)
    is smoother in time than the rotational disturbance (gravity gradient +
    target tumble), so a wider RBF kernel and faster forgetting suit translation.

    Inputs (translational): position-LOS error + LOS rate (6-vector).
    Outputs (translational): predicted disturbance in LOS frame (3 m/s²).

    Inputs (rotational): relative-quat vector part + relative ω body (6-vector).
    Outputs (rotational): predicted torque disturbance (3 N·m).
    """
    # Translational observer: wider kernel (sigma=2.0) — drag varies slowly.
    translational: RLSObserver = field(default_factory=lambda: RLSObserver(
        input_dim=6, output_dim=3, n_features=64, sigma=2.0, forgetting=0.95))
    # Rotational observer: tighter kernel — torque field has higher curvature
    # in (q_rv, ω_r) space because of the bilinear ω×Iω coupling.
    rotational: RLSObserver = field(default_factory=lambda: RLSObserver(
        input_dim=6, output_dim=3, n_features=64, sigma=1.0, forgetting=0.95))
    # Master switch: when False all predict() calls return 0 and update() is a
    # no-op. Lets a single controller object run with or without ML in tests.
    enabled: bool = True

    # Pre-update residuals from the most recent training call. Useful for
    # plotting convergence curves and choosing forgetting factors offline.
    last_residual_trans: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_residual_rot:   np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ------------------------------------------------------------------ predict
    def predict_translational(self, e_p: np.ndarray, e_p_dot: np.ndarray) -> np.ndarray:
        """Predicted LOS-frame disturbance for the current (e_p, ė_p) state."""
        if not self.enabled:
            return np.zeros(3)
        # Concatenate error + error-rate into the 6-D regression input.
        x = np.concatenate([e_p, e_p_dot])
        return self.translational.predict(x)

    def predict_rotational(self, q_rv: np.ndarray, omega_r: np.ndarray) -> np.ndarray:
        """Predicted body-frame disturbance torque from (q_rv, ω_r) state."""
        if not self.enabled:
            return np.zeros(3)
        x = np.concatenate([q_rv, omega_r])
        return self.rotational.predict(x)

    # ------------------------------------------------------------------- update
    def update_translational(self, e_p: np.ndarray, e_p_dot: np.ndarray,
                              measured_disturbance_los: np.ndarray) -> None:
        """Train translational RLS with the post-step disturbance residual."""
        if not self.enabled:
            return
        x = np.concatenate([e_p, e_p_dot])
        self.last_residual_trans = self.translational.update(x, measured_disturbance_los)

    def update_rotational(self, q_rv: np.ndarray, omega_r: np.ndarray,
                          measured_disturbance_torque_body: np.ndarray) -> None:
        """Train rotational RLS with the post-step torque-residual estimate."""
        if not self.enabled:
            return
        x = np.concatenate([q_rv, omega_r])
        self.last_residual_rot = self.rotational.update(x, measured_disturbance_torque_body)
