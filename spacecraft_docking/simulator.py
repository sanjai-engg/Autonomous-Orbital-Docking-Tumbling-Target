"""Simulation orchestrator with ZOH control + scipy.solve_ivp.

Implements the closed-loop architecture from §2 of the paper: each control
step samples the (noisy) state, computes thrust + torque, holds them
constant, and integrates the plant for dt_ctrl. Termination tolerances
follow Appendix D.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
from . import dynamics as dyn
from . import frames


@dataclass
class TrajectoryLog:
    """Per-step record of state, control, sliding surfaces, LOS angles."""
    rows: list = field(default_factory=list)
    success: bool = False
    final_time: float = 0.0

    def append(self, t: float, y: np.ndarray, u: np.ndarray, tau: np.ndarray, info: dict):
        """Capture one ZOH-step snapshot. Copies arrays to avoid aliasing."""
        self.rows.append({'t': t, 'y': y.copy(), 'u': u.copy(), 'tau': tau.copy(),
                          'info': {k: (v.copy() if isinstance(v, np.ndarray) else v)
                                   for k, v in info.items()}})

    def to_arrays(self) -> dict:
        """Stack rows into numpy arrays keyed by signal name (for plotting/CSV)."""
        n = len(self.rows)
        out = {
            't': np.array([r['t'] for r in self.rows]),
            'y': np.array([r['y'] for r in self.rows]),
            'u': np.array([r['u'] for r in self.rows]),
            'tau': np.array([r['tau'] for r in self.rows]),
            's_p': np.array([r['info']['s_p'] for r in self.rows]),
            's_a': np.array([r['info']['s_a'] for r in self.rows]),
            'r': np.array([r['info']['r'] for r in self.rows]),
            'alpha': np.array([r['info']['alpha'] for r in self.rows]),
            'beta': np.array([r['info']['beta'] for r in self.rows]),
            'alpha_t': np.array([r['info']['alpha_t'] for r in self.rows]),
            'beta_t': np.array([r['info']['beta_t'] for r in self.rows]),
        }
        return out


# Termination tolerances from paper Appendix D
LATERAL_TOL_M = 0.01        # max lateral port misalignment
AXIAL_TOL_M = 0.01          # max axial port gap
ATT_TOL_DEG = 1.0           # max relative-attitude Euler angle
LAT_VEL_TOL_MPS = 0.01      # max lateral closing-velocity component
ANG_VEL_TOL_DPS = 0.1       # max relative angular rate


def _initial_state(scenario, chaser, target) -> np.ndarray:
    """Build the 20-vector initial condition from a Scenario (Tables 1-3)."""
    pos = scenario.initial_pos_rel_m.copy()
    vel = scenario.initial_vel_rel_mps.copy()
    q_c = frames.euler_to_quat(np.deg2rad(scenario.initial_euler_chaser_deg))
    q_t = frames.euler_to_quat(np.deg2rad(scenario.initial_euler_target_deg))
    omega_c = np.deg2rad(scenario.initial_omega_chaser_body_dps).copy()
    omega_t = np.deg2rad(scenario.initial_omega_target_body_dps).copy()
    return dyn.pack_state(pos, vel, q_c, omega_c, q_t, omega_t)


def termination_met(y: np.ndarray, l_c: float, l_t: float) -> bool:
    """Successful-docking check based on port-to-port displacement (paper Appendix D, Eq. 25).

    Computes both ports' positions and velocities in LVLH, decomposes the
    port gap and the port-relative velocity into axial (along target +y_b)
    and lateral components, and checks all five Appendix-D tolerances.
    Velocity is port-relative — for a tumbling target the chaser rides
    around with it, so absolute LVLH chaser velocity is non-zero even
    when docking is achieved.
    """
    pos, vel, q_c, omega_c, q_t, omega_t = dyn.unpack_state(y)
    R_lvlh_chaser = frames.quat_to_dcm(q_c)
    R_lvlh_target = frames.quat_to_dcm(q_t)

    chaser_port_offset_body = np.array([0.0, -l_c, 0.0])
    target_port_offset_body = np.array([0.0,  l_t, 0.0])

    chaser_port_lvlh = pos + R_lvlh_chaser @ chaser_port_offset_body
    target_port_lvlh = R_lvlh_target @ target_port_offset_body
    d = chaser_port_lvlh - target_port_lvlh

    chaser_port_vel = vel + R_lvlh_chaser @ np.cross(omega_c, chaser_port_offset_body)
    target_port_vel = R_lvlh_target @ np.cross(omega_t, target_port_offset_body)
    v_rel = chaser_port_vel - target_port_vel

    axis = R_lvlh_target @ np.array([0.0, 1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    d_axial = float(np.dot(d, axis))
    d_lateral = float(np.linalg.norm(d - d_axial * axis))

    v_axial = float(np.dot(v_rel, axis))
    v_lat_norm = float(np.linalg.norm(v_rel - v_axial * axis))

    q_r = frames.quat_multiply(frames.quat_conjugate(q_t), q_c)
    euler_r_deg = np.rad2deg(frames.quat_to_euler(q_r))

    R_bc_bt = frames.quat_to_dcm(q_r)
    omega_r = omega_c - R_bc_bt @ omega_t
    omega_r_dps = np.rad2deg(omega_r)

    return (
        d_lateral < LATERAL_TOL_M
        and abs(d_axial) < AXIAL_TOL_M
        and np.all(np.abs(euler_r_deg) < ATT_TOL_DEG)
        and v_lat_norm < LAT_VEL_TOL_MPS
        and np.all(np.abs(omega_r_dps) < ANG_VEL_TOL_DPS)
    )


def _train_observer(controller, plant, t, y, y_next, u_lvlh, tau_body, info, dt):
    """Update the attached DisturbanceObserver from one ZOH step's residual.

    Operating principle
    -------------------
    During a ZOH step, the plant evolved from y(t) to y(t+dt) under the
    constant control inputs (u_lvlh, tau_body) and the actual environmental
    disturbances. We can recover what the disturbance must have been by
    reconciling Newton's second law over the step:

        I_c·ω̇_actual = τ_commanded + τ_gravity_gradient − ω×Iω + τ_unmodelled
        ⇒ τ_unmodelled = I_c·ω̇_actual + ω×Iω − τ_commanded − τ_gravity_gradient

    and similarly for the translational channel. The unmodelled component is
    the regression target; the regression input is whichever feature vector
    the controller used to predict (so train and predict are aligned).

    Channels
    --------
    Translational: project the achieved Cartesian acceleration into the LOS
    frame the controller works in, subtract the plant's modelled acceleration
    (control + gravity asymmetry) but add back the modelled environmental
    disturbance — the observer's job is to learn whatever is *left over*
    after every model term we already account for.

    Rotational: identical structure in chaser body frame, with τ_grav as the
    "known" piece and tau_body as the commanded piece.
    """
    pos, vel, q_c, omega_c, q_t, omega_t = dyn.unpack_state(y)
    pos_n, vel_n, q_c_n, omega_c_n, q_t_n, omega_t_n = dyn.unpack_state(y_next)

    # --- translational residual (LVLH then mapped to LOS) -----------------
    # Step 1: achieved chaser acceleration via first-difference of velocity.
    # Over a 50 ms ZOH step this is a good estimate of the average ÿ.
    accel_lvlh = (vel_n - vel) / dt
    # Step 2: nominal model expectation. Re-evaluate the plant derivative
    # *with* the same control input — this gives ÿ that the model believes
    # should have happened. Whatever the actual achieved minus the modelled
    # equals is the disturbance the model didn't capture.
    pos_dot, accel_nom_lvlh, *_ = dyn.unpack_state(
        plant.derivative(t, y, u_lvlh, tau_body))
    # Step 3: form the residual. Add back the modelled environmental
    # disturbance so the observer learns the true environmental effect
    # rather than the discrepancy with our (already-modelled) drag/J2.
    d_lvlh = accel_lvlh - accel_nom_lvlh + plant.disturbances.translational(t, pos, vel)
    # Step 4: rotate residual into the LOS frame, since the controller
    # subtracts d̂ from u_eq^LOS and the regression must learn the function
    # in the same coordinates.
    T_LS = info['T_LS']
    d_los = T_LS @ d_lvlh
    controller.observer.update_translational(info['e_p'], info['e_p_dot'], d_los)

    # --- rotational residual (chaser body frame) --------------------------
    # Step 1: achieved angular acceleration via first-difference of ω.
    omega_c_dot = (omega_c_n - omega_c) / dt
    I_c = controller.chaser_inertia
    # Step 2: known (modelled) gravity-gradient torque.
    tau_grav = plant.disturbances.gravity_gradient(q_c, which='chaser')
    # Step 3: solve Euler's equation backwards for total torque experienced:
    #   τ_total = I_c·ω̇ + ω × (I_c ω)
    tau_pred = I_c @ omega_c_dot + np.cross(omega_c, I_c @ omega_c)
    # Step 4: subtract the parts we already model — commanded torque and
    # gravity-gradient torque — leaving the unmodelled torque residual.
    d_tau = tau_pred - tau_body - tau_grav
    controller.observer.update_rotational(info['q_rv'], info['omega_r'], d_tau)


def simulate(scenario, plant, controller, sensors,
             dt_ctrl: float = 0.05, t_max_override: float | None = None) -> TrajectoryLog:
    """Closed-loop ZOH simulation. Returns a populated TrajectoryLog.

    Each iteration: measure → compute control → log → check termination →
    solve_ivp over dt_ctrl with thrust/torque held constant → renormalize
    quaternions to fight integration drift.
    """
    from .scenarios import CHASER, TARGET
    log = TrajectoryLog()
    y = _initial_state(scenario, CHASER, TARGET)
    t = 0.0
    t_max = t_max_override if t_max_override is not None else scenario.duration_s

    while t < t_max - 1e-12:
        meas = sensors.measure(y)
        m_pos, m_vel, m_qc, m_wc, m_qt, m_wt = dyn.unpack_state(meas)
        u_body, tau_body, info = controller.compute(t, m_pos, m_vel, m_qc, m_wc, m_qt, m_wt)
        log.append(t, y, u_body, tau_body, info)

        if termination_met(y, l_c=CHASER.docking_port_offset,
                           l_t=TARGET.docking_port_offset):
            log.success = True
            log.final_time = t
            return log

        # Plant expects thrust acceleration in LVLH; rotate from chaser body.
        q_c = y[6:10]
        u_lvlh = frames.quat_to_dcm(q_c) @ u_body
        sol = solve_ivp(
            plant.derivative,
            (t, t + dt_ctrl),
            y,
            args=(u_lvlh, tau_body),
            method='RK45', rtol=1e-8, atol=1e-10, max_step=dt_ctrl,
        )
        y_next = sol.y[:, -1]
        y_next[6:10]   = frames.quat_normalize(y_next[6:10])
        y_next[13:17]  = frames.quat_normalize(y_next[13:17])

        # Disturbance-observer training: residual between achieved acceleration
        # over the ZOH step and the plant's nominal expectation given (u, τ).
        # Residual treated as the unmodelled disturbance the observer must learn.
        if controller.observer is not None and controller.observer.enabled:
            _train_observer(controller, plant, t, y, y_next, u_lvlh, tau_body, info, dt_ctrl)

        y = y_next
        t += dt_ctrl

    log.final_time = t
    return log
