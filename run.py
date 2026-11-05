"""CLI entry point — run one or all scenarios, save figures and CSV.

Mirrors the paper's experimental section: 4 scenarios, each producing a
4-figure panel + a CSV trajectory log under outputs/.
"""
from __future__ import annotations
import argparse
import os
import sys
import csv
import numpy as np

from spacecraft_docking import scenarios as scen
from spacecraft_docking.disturbances import Disturbances
from spacecraft_docking.dynamics import Plant
from spacecraft_docking.control import SlidingModeController, ControlParams
from spacecraft_docking.sensors import SensorModel
from spacecraft_docking.simulator import simulate
from spacecraft_docking import plotting


# Paper figure numbering: scenario N → 4 figures starting at FIG_NUMBERS[N][0]
FIG_NUMBERS = {
    1: (3, 4, 5, 6),
    2: (7, 8, 9, 10),
    3: (11, 12, 13, 14),
    4: (15, 16, 17, 18),
}


def build_world(scenario, seed: int, no_noise: bool, observer: bool = False):
    """Wire up disturbances + plant + controller + sensors for one scenario.

    observer: when True, attach an RLS+RFF DisturbanceObserver and use the
    reduced reaching-law gains from ControlParams.observer_aided().
    """
    from spacecraft_docking.disturbance_observer import DisturbanceObserver
    dist = Disturbances(
        chaser_mass=scen.CHASER.mass, chaser_area=scen.CHASER.cross_section_m2,
        chaser_C_D=scen.CHASER.C_D, target_mass=scen.TARGET.mass,
        target_area=scen.TARGET.cross_section_m2, target_C_D=scen.TARGET.C_D,
        chaser_inertia=scen.CHASER.inertia, target_inertia=scen.TARGET.inertia,
        mu=scen.MU, R_E=scen.R_E, J2=scen.J2, orbit_altitude_m=scen.ORBIT_ALTITUDE_M,
    )
    plant = Plant(scen.CHASER.mass, scen.CHASER.inertia, scen.TARGET.inertia,
                  dist, scenario.external_torque_target)
    params = ControlParams.observer_aided() if observer else ControlParams.default()
    obs = DisturbanceObserver() if observer else None
    controller = SlidingModeController(scen.CHASER.mass, scen.CHASER.inertia,
                                        scen.TARGET.docking_port_offset,
                                        params,
                                        chaser_dock_offset=scen.CHASER.docking_port_offset,
                                        observer=obs)
    sensors = SensorModel(rng=np.random.default_rng(seed), enabled=not no_noise)
    return plant, controller, sensors


def write_csv(arr: dict, out_path: str) -> None:
    """Flatten the trajectory log into a single CSV (one row per ZOH step)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = ['t',
               'x', 'y', 'z', 'xdot', 'ydot', 'zdot',
               'qc0', 'qc1', 'qc2', 'qc3',
               'wcx', 'wcy', 'wcz',
               'qt0', 'qt1', 'qt2', 'qt3',
               'wtx', 'wty', 'wtz',
               'ux', 'uy', 'uz',
               'taux', 'tauy', 'tauz',
               'r', 'alpha', 'beta',
               'sp1', 'sp2', 'sp3',
               'sa1', 'sa2', 'sa3']
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(len(arr['t'])):
            w.writerow([
                arr['t'][i],
                *arr['y'][i, 0:3],
                *arr['y'][i, 3:6],
                *arr['y'][i, 6:10],
                *arr['y'][i, 10:13],
                *arr['y'][i, 13:17],
                *arr['y'][i, 17:20],
                *arr['u'][i],
                *arr['tau'][i],
                arr['r'][i], arr['alpha'][i], arr['beta'][i],
                *arr['s_p'][i],
                *arr['s_a'][i],
            ])


def run_scenario(num: int, args) -> bool:
    """Simulate one scenario, write 4 figures + CSV, optionally launch viewer."""
    scenario = scen.SCENARIOS[num]
    plant, controller, sensors = build_world(scenario, args.seed, args.no_noise,
                                              observer=args.observer)
    log = simulate(scenario, plant, controller, sensors)
    arr = log.to_arrays()

    out_dir = args.output_dir
    fig_dir = os.path.join(out_dir, 'figs', f'scenario{num}')
    fA, fB, fC, fD = FIG_NUMBERS[num]

    if not args.no_plots:
        plotting.plot_position_los(arr, os.path.join(fig_dir, f'fig{fA}_position_los.png'))
        plotting.plot_control_accel(arr, os.path.join(fig_dir, f'fig{fB}_control_accel.png'))
        plotting.plot_attitude_torque(arr, os.path.join(fig_dir, f'fig{fC}_attitude_torque.png'))
        plotting.plot_3d_trajectory(arr, scen.CHASER.dimensions, scen.TARGET.dimensions,
                                     os.path.join(fig_dir, f'fig{fD}_3d_trajectory.png'))

    if args.video:
        video_path = os.path.join(out_dir, 'video', f'scenario{num}.mp4')
        plotting.record_trajectory(arr, scen.CHASER.dimensions, scen.TARGET.dimensions,
                                   video_path, fps=args.fps, speed=args.video_speed,
                                   orbital_drift_mps=args.orbital_drift)
        print(f'  video → {video_path}')

    write_csv(arr, os.path.join(out_dir, 'data', f'scenario{num}.csv'))

    if args.interactive and num == args.scenarios_for_interactive:
        from spacecraft_docking.viewer import view_trajectory
        view_trajectory(arr, scen.CHASER.dimensions, scen.TARGET.dimensions)

    print(f'Scenario {num} ({scenario.name}): success={log.success}, '
          f'final_t={log.final_time:.2f}s, samples={len(arr["t"])}')
    return log.success


def main(argv=None):
    """Argparse + dispatch. Returns 0 on all-scenarios success, 1 otherwise."""
    p = argparse.ArgumentParser(description='SMC orbital docking simulator')
    p.add_argument('--scenario', required=True,
                   help='1, 2, 3, 4, or all')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-noise', action='store_true')
    p.add_argument('--observer', action='store_true',
                   help='Attach RLS+RFF disturbance observer; use reduced reaching-law gains')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--interactive', action='store_true',
                   help='Launch interactive viewer (single scenario only)')
    p.add_argument('--video', action='store_true',
                   help='Record MP4 of 3D trajectory to outputs/video/scenarioN.mp4 (needs ffmpeg)')
    p.add_argument('--fps', type=int, default=30, help='Video frame rate')
    p.add_argument('--video-speed', type=float, default=2.0,
                   help='Sim seconds per real second in video (e.g. 2.0 = 2× wall-clock)')
    p.add_argument('--orbital-drift', type=float, default=0.0,
                   help='Visual drift of target along +y_LVLH (m/s). LVLH is target-fixed, '
                        'so this is visualization-only; try 1.0 to see both craft moving')
    p.add_argument('--output-dir', default='outputs')
    args = p.parse_args(argv)

    if args.scenario == 'all':
        nums = [1, 2, 3, 4]
    else:
        nums = [int(args.scenario)]

    if args.interactive and len(nums) != 1:
        print('--interactive requires a single scenario.', file=sys.stderr)
        return 2
    args.scenarios_for_interactive = nums[0] if nums else None

    all_ok = True
    for n in nums:
        ok = run_scenario(n, args)
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
