import numpy as np
import pytest
from spacecraft_docking import sensors, dynamics


def test_sensor_zero_when_disabled():
    sm = sensors.SensorModel(rng=np.random.default_rng(0), enabled=False)
    pos = np.array([3.0, 6.0, 7.0])
    vel = np.zeros(3)
    qc = np.array([1.0, 0, 0, 0])
    wc = np.zeros(3)
    qt = np.array([1.0, 0, 0, 0])
    wt = np.zeros(3)
    y = dynamics.pack_state(pos, vel, qc, wc, qt, wt)
    out = sm.measure(y)
    np.testing.assert_array_equal(out, y)


def test_sensor_perturbs_and_normalises_quaternion():
    sm = sensors.SensorModel(rng=np.random.default_rng(0), enabled=True)
    pos = np.array([3.0, 6.0, 7.0])
    vel = np.zeros(3)
    qc = np.array([1.0, 0, 0, 0])
    wc = np.zeros(3)
    qt = np.array([1.0, 0, 0, 0])
    wt = np.zeros(3)
    y = dynamics.pack_state(pos, vel, qc, wc, qt, wt)
    out = sm.measure(y)
    qc_out = out[6:10]
    qt_out = out[13:17]
    assert np.linalg.norm(qc_out) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(qt_out) == pytest.approx(1.0, abs=1e-12)
    assert not np.allclose(out, y)
