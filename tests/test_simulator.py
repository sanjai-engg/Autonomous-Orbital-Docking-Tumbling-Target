import numpy as np
import pytest
from spacecraft_docking import simulator


def test_log_append_and_arrays():
    log = simulator.TrajectoryLog()
    y = np.zeros(20)
    log.append(t=0.0, y=y, u=np.zeros(3), tau=np.zeros(3),
               info={'s_p': np.zeros(3), 's_a': np.zeros(3),
                     'r': 0.0, 'alpha': 0.0, 'beta': 0.0,
                     'alpha_t': 0.0, 'beta_t': 0.0})
    log.append(t=0.05, y=y, u=np.zeros(3), tau=np.zeros(3),
               info={'s_p': np.zeros(3), 's_a': np.zeros(3),
                     'r': 0.0, 'alpha': 0.0, 'beta': 0.0,
                     'alpha_t': 0.0, 'beta_t': 0.0})
    arr = log.to_arrays()
    assert arr['t'].shape == (2,)
    assert arr['y'].shape == (2, 20)
    assert arr['u'].shape == (2, 3)


def test_termination_zero_state_passes():
    pos = np.array([0.0, 1.1 + 1.5, 0.0])
    vel = np.zeros(3)
    qc = np.array([1.0, 0, 0, 0])
    qt = np.array([1.0, 0, 0, 0])
    wc = np.zeros(3)
    wt = np.zeros(3)
    y = np.concatenate([pos, vel, qc, wc, qt, wt])
    assert simulator.termination_met(y, l_c=1.1, l_t=1.5)


def test_termination_far_fails():
    pos = np.array([3.0, 6.0, 7.0])
    vel = np.zeros(3)
    qc = np.array([1.0, 0, 0, 0])
    qt = np.array([1.0, 0, 0, 0])
    wc = np.zeros(3)
    wt = np.zeros(3)
    y = np.concatenate([pos, vel, qc, wc, qt, wt])
    assert not simulator.termination_met(y, l_c=1.1, l_t=1.5)


def test_simulate_runs_short():
    from spacecraft_docking import scenarios as scen
    from spacecraft_docking.disturbances import Disturbances
    from spacecraft_docking.dynamics import Plant
    from spacecraft_docking.control import SlidingModeController, ControlParams
    from spacecraft_docking.sensors import SensorModel

    s = scen.SCENARIOS[1]
    dist = Disturbances(
        chaser_mass=scen.CHASER.mass, chaser_area=scen.CHASER.cross_section_m2,
        chaser_C_D=scen.CHASER.C_D, target_mass=scen.TARGET.mass,
        target_area=scen.TARGET.cross_section_m2, target_C_D=scen.TARGET.C_D,
        chaser_inertia=scen.CHASER.inertia, target_inertia=scen.TARGET.inertia,
        mu=scen.MU, R_E=scen.R_E, J2=scen.J2, orbit_altitude_m=scen.ORBIT_ALTITUDE_M,
    )
    plant = Plant(scen.CHASER.mass, scen.CHASER.inertia, scen.TARGET.inertia,
                  dist, s.external_torque_target)
    controller = SlidingModeController(scen.CHASER.mass, scen.CHASER.inertia,
                                        scen.TARGET.docking_port_offset,
                                        ControlParams.default(),
                                        chaser_dock_offset=scen.CHASER.docking_port_offset)
    sm = SensorModel(rng=np.random.default_rng(0), enabled=False)

    log = simulator.simulate(s, plant, controller, sm,
                             dt_ctrl=0.05, t_max_override=2.0)
    arr = log.to_arrays()
    assert arr['t'][-1] >= 1.5
    assert np.all(np.isfinite(arr['y']))
