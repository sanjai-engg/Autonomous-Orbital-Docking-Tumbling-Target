"""Sliding-mode controller for translational + rotational docking.

Paper §4 — full SMC formulation. Translational channel uses LOS-frame
spherical errors (r, α', β'); rotational channel uses relative quaternion.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import frames
from .disturbance_observer import DisturbanceObserver


def sat(s: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Boundary-layer saturation (paper Eq. 61) — replaces sign(s) to suppress chatter."""
    return np.clip(s / eps, -1.0, 1.0)


def relative_quaternion(q_chaser: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """q_r = q_target⁻¹ ⊗ q_chaser. Small vector part when frames are aligned."""
    return frames.quat_multiply(frames.quat_conjugate(q_target), q_chaser)


@dataclass(frozen=True)
class ControlParams:
    """SMC gains and saturation limits (paper Table C2 default values)."""
    lambda_p: np.ndarray
    lambda_a: np.ndarray
    eta_p: np.ndarray
    eta_a: np.ndarray
    eps_p: np.ndarray
    eps_a: np.ndarray
    k_p: np.ndarray
    k_a: np.ndarray
    u_max: float = 0.5
    tau_max: float = 1.5

    @staticmethod
    def default() -> 'ControlParams':
        """Paper Table C2 base gains, with tightened boundary layers (ε_p, ε_a).

        Plan-spec ε values produce >1° steady-state error in scenarios 1 and 4
        within their published 50/70 s durations. Reduced ε to 0.01 (translational)
        and 0.005 (rotational) so all four scenarios meet Appendix-D termination
        within scenario duration.
        """
        return ControlParams(
            lambda_p=np.array([0.15, 0.3, 0.2]),
            lambda_a=np.array([0.3, 0.3, 0.3]),
            eta_p=np.array([0.01, 0.01, 0.01]),
            eta_a=np.array([0.02, 0.02, 0.012]),
            eps_p=np.array([0.01, 0.01, 0.01]),
            eps_a=np.array([0.005, 0.005, 0.005]),
            k_p=np.array([0.2, 0.2, 0.2]),
            k_a=np.array([0.5, 0.7, 0.9]),
        )

    @staticmethod
    def observer_aided() -> 'ControlParams':
        """Reduced reaching-law gains, intended for use with DisturbanceObserver.

        With the observer feeding forward an estimate of the unmodeled
        disturbance, the SMC reaching gains k_p and k_a can be cut without
        losing robustness. Lower k → narrower 'switching' amplitude → less
        chatter on commanded thrust and torque.

        Empirically, scaling k by 0.7 keeps Appendix-D termination intact in
        all four scenarios while reducing thrust chatter by 40-56% and torque
        chatter by 9-32%; see scripts/compare_observer.py for the sweep that
        established this factor.
        """
        base = ControlParams.default()
        return ControlParams(
            # Lambda, eta, eps inherited unchanged — only k is reduced.
            lambda_p=base.lambda_p, lambda_a=base.lambda_a,
            eta_p=base.eta_p, eta_a=base.eta_a,
            eps_p=base.eps_p, eps_a=base.eps_a,
            k_p=0.7 * base.k_p,
            k_a=0.7 * base.k_a,
            u_max=base.u_max, tau_max=base.tau_max,
        )


@dataclass
class SlidingModeController:
    """SMC orchestrator: produces chaser thrust + body torque each control step.

    Optional DisturbanceObserver feeds forward an online-learned estimate of
    unmodelled disturbances. When attached, the predicted disturbance is
    subtracted from u_eq / tau_eq, allowing the reaching-law gains to be cut
    (use ControlParams.observer_aided()) for less chatter.
    """
    chaser_mass: float
    chaser_inertia: np.ndarray
    target_dock_offset: float
    params: ControlParams
    chaser_dock_offset: float = 1.1     # paper Table 1 default; set to actual l_c at construction
    observer: DisturbanceObserver | None = None

    @property
    def desired_range_m(self) -> float:
        """r_d = l_c + l_t — radial distance at which dock ports meet."""
        return self.chaser_dock_offset + self.target_dock_offset

    def translational_control(self, los: np.ndarray, los_dot: np.ndarray,
                              alpha_t: float, beta_t: float,
                              alpha_t_dot: float = 0.0, beta_t_dot: float = 0.0):
        """Translational SMC in LOS frame. Returns (u_LOS, sliding-surface s_p).

        Implements paper Eqs 26, 28-30, 37, 43, 46, 49: equivalent control u_eq
        that holds s_p=0 plus reaching law u_s = -k_p · sat(s_p/ε_p).
        Tumbling target: pass nonzero α̇_t, β̇_t to feed-forward port-angle rates.
        """
        r, alpha, beta = los
        rdot, alpha_dot, beta_dot = los_dot
        ap = alpha - alpha_t
        bp = beta - beta_t
        ap_dot = alpha_dot - alpha_t_dot
        bp_dot = beta_dot - beta_t_dot

        # Radial error: drive r to (l_c + l_t), not zero — that puts dock ports together.
        e_p = np.array([r - self.desired_range_m, ap, bp])
        e_p_dot = np.array([rdot, ap_dot, bp_dot])
        s_p = e_p_dot + self.params.lambda_p * e_p

        u_r_eq  = -r * bp_dot**2 * np.cos(ap)**2 - r * ap_dot**2 \
                  - self.params.lambda_p[0] * rdot
        u_ap_eq = 2 * rdot * ap_dot + r * bp_dot**2 * np.sin(ap) * np.cos(ap) \
                  - r * self.params.lambda_p[1] * ap_dot
        u_bp_eq = 2 * rdot * bp_dot * np.cos(ap) \
                  - 2 * r * bp_dot * ap_dot * np.sin(ap) \
                  - r * self.params.lambda_p[2] * bp_dot * np.cos(ap)
        u_eq = np.array([u_r_eq, u_ap_eq, u_bp_eq])

        # Observer feed-forward.
        # If a DisturbanceObserver is attached, query it for the predicted
        # residual disturbance d̂_p(e_p, e_p_dot) in the LOS frame and subtract
        # it from the equivalent control. This is the "feed-forward
        # cancellation" half of the observer-aided SMC: anything the observer
        # already accounts for need not be absorbed by the discontinuous
        # reaching law, which is why ControlParams.observer_aided() can use a
        # smaller k_p without losing robustness.
        d_hat = (self.observer.predict_translational(e_p, e_p_dot)
                 if self.observer is not None else np.zeros(3))

        # Reaching law: -k_p · sat(s_p / eps_p) — the discontinuous correction
        # that drives the state onto the sliding surface. Boundary-layer sat()
        # replaces sgn() to suppress chatter at the cost of a small steady-
        # state error within the layer width eps_p.
        u_s = -self.params.k_p * sat(s_p, self.params.eps_p)
        u_los = u_eq + u_s - d_hat

        u_los = np.clip(u_los, -self.params.u_max, self.params.u_max)
        return u_los, s_p

    def rotational_control(self, q_c: np.ndarray, omega_c: np.ndarray,
                           q_t: np.ndarray, omega_t: np.ndarray):
        """Rotational SMC. Returns (τ_body, sliding-surface s_a).

        Paper Eqs 26, 51-54, 56, 60. Sliding surface defined on relative-
        quaternion vector part. Q_r is the kinematic Jacobian from Eq. 52.
        """
        q_r = relative_quaternion(q_c, q_t)
        q_r0, q_rv = q_r[0], q_r[1:4]

        R_bc_bt = frames.quat_to_dcm(q_r)
        omega_r = omega_c - R_bc_bt @ omega_t

        Q_r = np.array([
            [q_r[0], -q_r[3],  q_r[2]],
            [q_r[3],  q_r[0], -q_r[1]],
            [-q_r[2], q_r[1],  q_r[0]],
        ])
        q_rv_dot = 0.5 * (Q_r @ omega_r)

        s_a = q_rv_dot + self.params.lambda_a * q_rv

        I_c = self.chaser_inertia
        f_omega = (
            -np.linalg.solve(I_c, np.cross(omega_c, I_c @ omega_c))
            + np.cross(omega_c, R_bc_bt @ omega_t)
        )

        Q_r_inv = np.linalg.inv(Q_r)
        tau_eq = 2 * I_c @ Q_r_inv @ (
            -0.5 * Q_r @ f_omega
            - 0.5 * (q_r0 * omega_r - np.cross(omega_r, q_rv))
            - self.params.lambda_a * q_rv_dot
        )

        # Observer feed-forward (rotational channel).
        # Mirror of the translational case: the rotational observer returns a
        # body-frame torque estimate that captures whatever the model misses
        # (target tumble coupling, gravity-gradient bias, sensor mis-bias).
        # Subtracting it lightens the load on the discontinuous reaching law.
        tau_hat = (self.observer.predict_rotational(q_rv, omega_r)
                   if self.observer is not None else np.zeros(3))

        # Reaching law on the relative-quaternion sliding surface s_a.
        tau_s = -self.params.k_a * sat(s_a, self.params.eps_a)
        tau = tau_eq + tau_s - tau_hat
        tau = np.clip(tau, -self.params.tau_max, self.params.tau_max)
        return tau, s_a

    def compute(self, t: float,
                pos_rel: np.ndarray, vel_rel: np.ndarray,
                q_c: np.ndarray, omega_c: np.ndarray,
                q_t: np.ndarray, omega_t: np.ndarray):  # omega_t expected in target body frame

        """Full control computation called once per ZOH step.

        Pipeline:
          1. LVLH state → LOS spherical state (r, α, β) and rates (paper Eq. 3, §4.1).
          2. Target docking-port angles (Eq. 22).
          3. translational_control() in LOS, rotational_control() in body.
          4. Map LOS thrust → LVLH (T_LS.T) → chaser body (R_body→LVLH)^T.
        """
        r, alpha, beta = frames.lvlh_to_los(pos_rel)
        # LOS-rate kinematics — closed form from differentiating Eq. 3
        x, y, z = pos_rel
        xdot, ydot, zdot = vel_rel
        rho = np.sqrt(x*x + y*y)
        if r > 1e-9:
            r_dot = (x*xdot + y*ydot + z*zdot) / r
        else:
            r_dot = 0.0
        if rho > 1e-9:
            alpha_dot = (zdot * rho - z * (x*xdot + y*ydot) / rho) / (r*r)
            beta_dot  = (-xdot * y + x * ydot) / (rho*rho)
        else:
            alpha_dot = 0.0
            beta_dot = 0.0

        alpha_t, beta_t = frames.docking_port_angles(q_t, self.target_dock_offset)
        alpha_t_dot, beta_t_dot = frames.docking_port_angle_rates(
            q_t, omega_t, self.target_dock_offset)
        los = np.array([r, alpha, beta])
        los_dot = np.array([r_dot, alpha_dot, beta_dot])

        u_los, s_p = self.translational_control(los, los_dot, alpha_t, beta_t,
                                                 alpha_t_dot, beta_t_dot)

        # Frame chain: LOS → LVLH (T_LS is orthogonal per Eq. 4) → chaser body
        T_LS_now = frames.los_jacobian(alpha, beta)
        u_lvlh = T_LS_now.T @ u_los
        R_lvlh_from_chaser = frames.quat_to_dcm(q_c)
        u_body = R_lvlh_from_chaser.T @ u_lvlh
        u_body = np.clip(u_body, -self.params.u_max, self.params.u_max)

        tau_body, s_a = self.rotational_control(q_c, omega_c, q_t, omega_t)

        # Persist regression features (the (x_p, x_a) inputs the controller
        # used) into `info` so the simulator can train the disturbance
        # observer on the matching feature vector after the ZOH step
        # completes. Keeping these here avoids re-deriving them in the
        # simulator and guarantees train/predict use exactly the same x.
        ap = alpha - alpha_t
        bp = beta - beta_t
        e_p     = np.array([r - self.desired_range_m, ap, bp])
        e_p_dot = np.array([r_dot, alpha_dot - alpha_t_dot, beta_dot - beta_t_dot])
        q_r = relative_quaternion(q_c, q_t)
        q_rv = q_r[1:4]
        R_bc_bt = frames.quat_to_dcm(q_r)
        omega_r = omega_c - R_bc_bt @ omega_t

        info = {
            's_p': s_p, 's_a': s_a,                         # sliding-surface diagnostics
            'r': r, 'alpha': alpha, 'beta': beta,           # raw LOS state (for plotting)
            'alpha_t': alpha_t, 'beta_t': beta_t,           # target-port angles
            'e_p': e_p, 'e_p_dot': e_p_dot,                 # translational regression input
            'q_rv': q_rv, 'omega_r': omega_r,               # rotational regression input
            'T_LS': T_LS_now,                               # LVLH→LOS rotation (cached)
        }
        return u_body, tau_body, info
