"""Interactive 3D trajectory viewer (matplotlib widgets).

Adds a time-slider and play/pause button on top of `plot_3d_trajectory`,
for inspecting docking dynamics step-by-step. Not used in batch runs.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from . import frames
from .plotting import draw_cube


def view_trajectory(arr: dict, chaser_dims: np.ndarray, target_dims: np.ndarray) -> None:
    """Open a window showing chaser + target poses scrubbable across the trajectory."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)

    t = arr['t']
    y = arr['y']
    pos = y[:, 0:3]
    qc = y[:, 6:10]
    qt = y[:, 13:17]

    line, = ax.plot([], [], [], 'g-', linewidth=1.5)

    def redraw(idx: int) -> None:
        ax.collections.clear()
        idx = int(idx)
        line.set_data(pos[:idx+1, 0], pos[:idx+1, 1])
        line.set_3d_properties(pos[:idx+1, 2])
        draw_cube(ax, pos[idx], qc[idx], chaser_dims, 'lightgray', alpha=0.6)
        draw_cube(ax, np.zeros(3), qt[idx], target_dims, 'blue', alpha=0.4)
        x_min, x_max = pos[:, 0].min()-1, pos[:, 0].max()+1
        y_min, y_max = pos[:, 1].min()-1, pos[:, 1].max()+1
        z_min, z_max = pos[:, 2].min()-1, pos[:, 2].max()+1
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
        ax.set_xlabel('x_LVLH'); ax.set_ylabel('y_LVLH'); ax.set_zlabel('z_LVLH')
        ax.set_title(f't = {t[idx]:.2f} s')
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.15, 0.06, 0.7, 0.03])
    slider = Slider(ax_slider, 'time', 0, len(t)-1, valinit=0, valstep=1)
    slider.on_changed(redraw)

    ax_play = plt.axes([0.05, 0.05, 0.08, 0.04])
    btn = Button(ax_play, 'Play')
    state = {'playing': False, 'idx': 0}

    def play(_event):
        state['playing'] = not state['playing']
        if state['playing']:
            timer = fig.canvas.new_timer(interval=50)
            timer.add_callback(advance)
            timer.start()
            state['timer'] = timer
        else:
            state['timer'].stop()

    def advance():
        if not state['playing']:
            return
        state['idx'] = (state['idx'] + 1) % len(t)
        slider.set_val(state['idx'])

    btn.on_clicked(play)
    redraw(0)
    plt.show()
