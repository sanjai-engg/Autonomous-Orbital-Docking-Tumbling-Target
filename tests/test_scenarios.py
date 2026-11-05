import numpy as np
import pytest

from spacecraft_docking import scenarios
from spacecraft_docking import scenarios as scen
from spacecraft_docking.disturbances import Disturbances
from spacecraft_docking.dynamics import Plant
from spacecraft_docking.control import SlidingModeController, ControlParams
from spacecraft_docking.sensors import SensorModel
from spacecraft_docking.simulator import simulate


def test_chaser_target_params():
    assert scenarios.CHASER.mass == 150.0
    assert scenarios.TARGET.mass == 400.0
    np.testing.assert_array_equal(scenarios.CHASER.dimensions, [1.2, 1.0, 0.8])
    assert scenarios.TARGET.docking_port_offset == 1.5


def test_four_scenarios_present():
    for k in (1, 2, 3, 4):
        assert k in scenarios.SCENARIOS
    s2 = scenarios.SCENARIOS[2]
    np.testing.assert_array_equal(s2.initial_omega_target_body_dps, [1.5, 0.5, 0.0])


def test_scenario3_torque_callable():
    s3 = scenarios.SCENARIOS[3]
    tau = s3.external_torque_target(0.0)
    assert tau.shape == (3,)
    np.testing.assert_allclose(tau, [0.0, 0.0, 0.0], atol=1e-12)
    tau_t = s3.external_torque_target(5.0)
    assert tau_t[0] != 0.0


def _run(num: int, no_noise: bool = True):
    s = scen.SCENARIOS[num]
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
    sensors_model = SensorModel(rng=np.random.default_rng(0), enabled=not no_noise)
    return simulate(s, plant, controller, sensors_model)


@pytest.mark.slow
def test_scenario1_terminates_successfully():
    log = _run(1, no_noise=True)
    assert log.success, f"Scenario 1 did not meet termination criteria; final_t={log.final_time}"


@pytest.mark.slow
def test_scenario2_terminates_successfully():
    log = _run(2, no_noise=True)
    assert log.success, f"Scenario 2 did not meet termination criteria; final_t={log.final_time}"


@pytest.mark.slow
def test_scenario3_terminates_successfully():
    log = _run(3, no_noise=True)
    assert log.success, f"Scenario 3 did not meet termination criteria; final_t={log.final_time}"


@pytest.mark.slow
def test_scenario4_terminates_successfully():
    log = _run(4, no_noise=True)
    assert log.success, f"Scenario 4 did not meet termination criteria; final_t={log.final_time}"
