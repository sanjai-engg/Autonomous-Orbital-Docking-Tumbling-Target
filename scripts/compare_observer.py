"""Quantify chatter reduction from the disturbance observer.

Runs each scenario twice — once with the default SMC, once with the
ML-disturbance-observer wrapper plus reduced reaching-law gains — and reports
two chatter metrics per channel:

  rms_du   = RMS of consecutive control-step differences in commanded thrust
             (m/s² per step). Lower means smoother thrust.
  rms_dtau = same metric for commanded torque (N·m per step).

Lower values indicate less chatter. The observer is expected to reduce both
without losing the Appendix-D termination success.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from spacecraft_docking import scenarios as scen
from spacecraft_docking.disturbances import Disturbances
from spacecraft_docking.dynamics import Plant
from spacecraft_docking.control import SlidingModeController, ControlParams
from spacecraft_docking.sensors import SensorModel
from spacecraft_docking.simulator import simulate
from spacecraft_docking.disturbance_observer import DisturbanceObserver


def _build(scenario, *, observer: bool):
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
    ctrl = SlidingModeController(scen.CHASER.mass, scen.CHASER.inertia,
                                  scen.TARGET.docking_port_offset, params,
                                  chaser_dock_offset=scen.CHASER.docking_port_offset,
                                  observer=obs)
    sm = SensorModel(rng=np.random.default_rng(0), enabled=False)
    return plant, ctrl, sm


def chatter_metric(arr, key: str) -> float:
    x = arr[key]
    return float(np.sqrt(np.mean(np.diff(x, axis=0) ** 2)))


def main():
    print(f"{'scen':<5}{'mode':<12}{'t_dock':>8}{'rms_du':>12}{'rms_dtau':>12}{'success':>10}")
    print('-' * 60)
    for sid in [1, 2, 3, 4]:
        s = scen.SCENARIOS[sid]
        for label, use_obs in [('baseline', False), ('observer', True)]:
            plant, ctrl, sm = _build(s, observer=use_obs)
            log = simulate(s, plant, ctrl, sm)
            arr = log.to_arrays()
            print(f'{sid:<5}{label:<12}{log.final_time:>8.2f}'
                  f'{chatter_metric(arr, "u"):>12.5f}'
                  f'{chatter_metric(arr, "tau"):>12.5f}'
                  f'{str(log.success):>10}')


if __name__ == '__main__':
    main()
