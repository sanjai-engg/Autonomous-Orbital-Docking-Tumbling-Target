"""Static plot generation matching paper figures.

Per scenario the paper publishes 4 figures: position+LOS, control accel,
attitude+torque, 3D trajectory. Mapping by scenario:
  Scenario 1 → Figs 3-6, Scenario 2 → 7-10, Scenario 3 → 11-14, Scenario 4 → 15-18.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import frames


def plot_position_los(arr: dict, out_path: str) -> None:
    """LOS errors α', β' and range r (top) plus their rates (bottom). Figs 3, 7, 11, 15."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 6))
    t = arr['t']
    r = arr['r']
    alpha_p_deg = np.rad2deg(arr['alpha'] - arr['alpha_t'])
    beta_p_deg  = np.rad2deg(arr['beta']  - arr['beta_t'])

    ax = axes[0]
    ax.plot(t, alpha_p_deg, 'r--', label="alpha'")
    ax.plot(t, beta_p_deg, 'k-.', label="beta'")
    ax2 = ax.twinx()
    ax2.plot(t, r, 'b-', label='r')
    ax.set_xlabel('time (s)')
    ax.set_ylabel("alpha', beta' (deg)")
    ax2.set_ylabel('r (m)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    rdot = np.gradient(r, t)
    apdot = np.gradient(alpha_p_deg, t)
    bpdot = np.gradient(beta_p_deg, t)

    ax = axes[1]
    ax.plot(t, apdot, 'r--', label="alpha_dot'")
    ax.plot(t, bpdot, 'k-.', label="beta_dot'")
    ax3 = ax.twinx()
    ax3.plot(t, rdot, 'b-', label='r_dot')
    ax.set_xlabel('time (s)')
    ax.set_ylabel("alpha_dot', beta_dot' (deg/s)")
    ax3.set_ylabel('r_dot (m/s)')
    ax.legend(loc='upper left')
    ax3.legend(loc='upper right')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_control_accel(arr: dict, out_path: str) -> None:
    """Body-frame thrust acceleration components and magnitude. Figs 4, 8, 12, 16."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 6))
    t = arr['t']
    u = arr['u']

    axes[0].plot(t, u[:, 0], 'b-',  label='u_x')
    axes[0].plot(t, u[:, 1], 'r--', label='u_y')
    axes[0].plot(t, u[:, 2], 'k-.', label='u_z')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('u (m/s^2) - body frame')
    axes[0].legend()

    axes[1].plot(t, np.linalg.norm(u, axis=1), 'b-')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('||u|| (m/s^2)')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_attitude_torque(arr: dict, out_path: str) -> None:
    """4-panel: relative quaternion, relative Euler, relative ω, applied torque. Figs 5, 9, 13, 17."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    t = arr['t']
    y = arr['y']
    qc = y[:, 6:10]
    qt = y[:, 13:17]
    omega_c = y[:, 10:13]
    omega_t = y[:, 17:20]

    qr = np.array([
        frames.quat_multiply(frames.quat_conjugate(qt[i]), qc[i])
        for i in range(len(t))
    ])
    eul_r_deg = np.rad2deg(np.array([frames.quat_to_euler(q) for q in qr]))

    ax = axes[0, 0]
    ax.plot(t, qr[:, 0], 'b-',  label='q_r0')
    ax.plot(t, qr[:, 1], 'r--', label='q_r1')
    ax.plot(t, qr[:, 2], 'k-.', label='q_r2')
    ax.plot(t, qr[:, 3], 'm:',  label='q_r3')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('q_r')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, eul_r_deg[:, 0], 'b-',  label='phi_r')
    ax.plot(t, eul_r_deg[:, 1], 'r--', label='theta_r')
    ax.plot(t, eul_r_deg[:, 2], 'k-.', label='psi_r')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Euler angles (deg)')
    ax.legend()

    omega_r = np.zeros_like(omega_c)
    for i in range(len(t)):
        R_bc_bt = frames.quat_to_dcm(qr[i])
        omega_r[i] = omega_c[i] - R_bc_bt @ omega_t[i]
    omega_r_dps = np.rad2deg(omega_r)

    ax = axes[1, 0]
    ax.plot(t, omega_r_dps[:, 0], 'b-',  label='omega_rx')
    ax.plot(t, omega_r_dps[:, 1], 'r--', label='omega_ry')
    ax.plot(t, omega_r_dps[:, 2], 'k-.', label='omega_rz')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('omega_r (deg/s)')
    ax.legend()

    tau = arr['tau']
    ax = axes[1, 1]
    ax.plot(t, tau[:, 0], 'b-',  label='tau_x')
    ax.plot(t, tau[:, 1], 'r--', label='tau_y')
    ax.plot(t, tau[:, 2], 'k-.', label='tau_z')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('tau (Nm)')
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _cube_corners(dimensions: np.ndarray) -> np.ndarray:
    """8 corner positions of an axis-aligned cuboid centered at origin."""
    Lx, Ly, Lz = dimensions
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]], float)
    return s * 0.5 * np.array([Lx, Ly, Lz])


_CUBE_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
               (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]


def draw_cube(ax, position: np.ndarray, q: np.ndarray, dimensions: np.ndarray,
              colour: str, alpha: float = 0.2,
              port_offset: float | None = None,
              port_colour: str = 'red') -> None:
    """Render a body-frame cuboid at LVLH position with attitude q on a 3D axis.

    If port_offset is set, also draws a line from COM along body +y_b of that
    length (positive) or along -y_b (negative). Makes attitude rotation visible.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    R = frames.quat_to_dcm(q)
    corners_body = _cube_corners(dimensions)
    corners_world = (R @ corners_body.T).T + position
    faces = [[corners_world[i] for i in f] for f in _CUBE_FACES]
    poly = Poly3DCollection(faces, alpha=alpha, edgecolor='k', linewidths=0.6)
    poly.set_facecolor(colour)
    ax.add_collection3d(poly)

    if port_offset is not None:
        port_world = position + R @ np.array([0.0, port_offset, 0.0])
        ax.plot([position[0], port_world[0]],
                [position[1], port_world[1]],
                [position[2], port_world[2]],
                color=port_colour, linewidth=2.5)
        ax.scatter([port_world[0]], [port_world[1]], [port_world[2]],
                   color=port_colour, s=30)


def record_trajectory(arr: dict, chaser_dims: np.ndarray, target_dims: np.ndarray,
                      out_path: str, fps: int = 30, speed: float = 2.0,
                      chaser_port_offset: float = 1.1,
                      target_port_offset: float = 1.5,
                      orbital_drift_mps: float = 0.0) -> None:
    """Render the docking trajectory as an MP4 (chaser + target cubes + COM trail).

    fps: output frame rate. speed: simulation-seconds per real-second
    (e.g., 2.0 = 2× faster than wall clock). Requires ffmpeg in PATH.

    orbital_drift_mps: visualization-only drift of the target along +y_LVLH
    (V-bar) at this synthetic velocity. LVLH is target-fixed, so target is
    stationary in physics; this drift makes orbital progression visible by
    translating the whole stage. Real orbital velocity is ~7.7 km/s — drowns
    the cubes — so use a small value like 0.5–2 m/s for visual clarity.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    t = arr['t']
    y = arr['y']
    pos_rel = y[:, 0:3]
    qc = y[:, 6:10]
    qt = y[:, 13:17]

    target_pos = np.zeros_like(pos_rel)
    target_pos[:, 1] = orbital_drift_mps * t
    chaser_pos = pos_rel + target_pos

    sim_dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
    real_dt = 1.0 / fps
    step = max(int(round(speed * real_dt / sim_dt)), 1)
    frames = list(range(0, len(t), step))
    if frames[-1] != len(t) - 1:
        frames.append(len(t) - 1)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(chaser_pos[:, 0], chaser_pos[:, 1], chaser_pos[:, 2],
            color='lightgreen', linewidth=0.6, alpha=0.5, label='chaser COM')
    if orbital_drift_mps > 0:
        ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
                color='lightblue', linewidth=0.6, alpha=0.5, label='target COM')
    pad = 1.5
    all_pos = np.vstack([chaser_pos, target_pos])
    ax.set_xlim(all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ax.set_ylim(all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)
    ax.set_zlim(all_pos[:, 2].min() - pad, all_pos[:, 2].max() + pad)
    ax.set_xlabel('x_LVLH (m)'); ax.set_ylabel('y_LVLH (m, V-bar)'); ax.set_zlabel('z_LVLH (m)')
    title = ax.set_title('')
    n_static = len(ax.lines)

    def update(i):
        for c in list(ax.collections):
            c.remove()
        for ln in list(ax.lines)[n_static:]:
            ln.remove()
        draw_cube(ax, chaser_pos[i], qc[i], chaser_dims, colour='lightgray', alpha=0.55,
                  port_offset=-chaser_port_offset, port_colour='orange')
        draw_cube(ax, target_pos[i], qt[i], target_dims, colour='steelblue', alpha=0.45,
                  port_offset=target_port_offset, port_colour='red')
        title.set_text(f't = {t[i]:.2f} s   r = {np.linalg.norm(pos_rel[i]):.2f} m')
        return ()

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2400)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)


def plot_3d_trajectory(arr: dict, chaser_dims: np.ndarray, target_dims: np.ndarray,
                        out_path: str, n_chaser_snapshots: int = 8) -> None:
    """3D LVLH trajectory with target + chaser snapshots overlaid. Figs 6, 10, 14, 18."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    t = arr['t']
    y = arr['y']
    pos = y[:, 0:3]
    qc = y[:, 6:10]
    qt = y[:, 13:17]

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'g-', linewidth=1.5)

    idx = np.linspace(0, len(t)-1, n_chaser_snapshots).astype(int)
    for i in idx:
        draw_cube(ax, pos[i], qc[i], chaser_dims, colour='lightgray', alpha=0.18)

    for i, alpha in [(0, 0.3), (len(t)//2, 0.2), (-1, 0.4)]:
        draw_cube(ax, np.zeros(3), qt[i], target_dims, colour='blue', alpha=alpha)

    ax.set_xlabel('x_LVLH (m)')
    ax.set_ylabel('y_LVLH (m)')
    ax.set_zlabel('z_LVLH (m)')
    ax.set_title('Docking trajectory')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
