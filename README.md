# Autonomous Orbital Docking with a Tumbling Target

A Python simulation study reproducing a published six-degree-of-freedom (6-DOF) sliding-mode controller for autonomous docking with a freely tumbling target spacecraft.

The implementation models relative translational motion in the target-centred local-vertical/local-horizontal (LVLH) frame, chaser and target attitude dynamics, environmental disturbances, and noisy state measurements. It also includes an analytic target docking-port angular-rate feed-forward term and an online RLS-RFF disturbance observer.

> **Scope:** This is a numerical simulation and controller-reproduction project. It is not a physical spacecraft docking demonstration or flight-qualified system.

## Project highlights

- Reproduced a 6-DOF chaser–target closed-loop model with translational and rotational sliding-mode control.
- Modelled atmospheric drag, differential (J_2) perturbation, gravity-gradient torque, and Gaussian sensor noise.
- Simulated four target-motion scenarios, all satisfying the implemented terminal docking gate.
- Added analytic feed-forward of target docking-port angular rates to the LOS tracking surface to compensate for target motion during the zero-order-hold (ZOH) interval.
- Implemented an online Recursive Least Squares with Random Fourier Features (RLS-RFF) disturbance observer for translational and rotational channels.
- Included plots, trajectory CSV files, and MP4 visualisations for the four scenarios.

## Repository structure

```text
Autonomous-Orbital-Docking-Tumbling-Target/
├── README.md
├── run.py
├── requirements.txt
├── pytest.ini
├── .gitignore
├── spacecraft_docking/
│   ├── frames.py
│   ├── dynamics.py
│   ├── disturbances.py
│   ├── control.py
│   ├── sensors.py
│   ├── scenarios.py
│   ├── simulator.py
│   ├── plotting.py
│   ├── viewer.py
│   └── disturbance_observer.py
├── scripts/
│   └── compare_observer.py
├── tests/
│   └── test_*.py
├── MATLAB/
│   └── EKF_Docking_Simulation/
│       ├── main_closedloop.m
│       ├── navigation_ekf.m
│       ├── attitude_dynamics.m
│       ├── cw_dynamics.m
│       ├── rk4_step.m
│       ├── guidance_smc.m
│       ├── attitude_smc.m
│       └── README.md
└── outputs/
    ├── figs/scenarioN/
    ├── data/
    └── video/
```

## Model and controller

The simulation state contains the relative position and velocity, chaser and target attitude quaternions, and their body angular velocities. The translational sliding-mode controller uses line-of-sight (LOS) range, elevation, and azimuth coordinates. The rotational controller uses the relative quaternion and relative angular velocity.

The closed loop uses a **50 ms control period (20 Hz)**. The plant is integrated across each ZOH interval with SciPy's RK45 solver. Chaser and target quaternions are renormalised after each integration step.

## Baseline docking results

The baseline implementation was run on all four scenarios. Each run returned a successful docking flag under the implemented terminal gate.

| Scenario | Target condition | Docking time |
|---:|---|---:|
| 1 | No initial tumble; no external torque | 46.55 s |
| 2 | Initial target angular velocity | 46.70 s |
| 3 | Sinusoidal external torque | 48.20 s |
| 4 | Initial tumble and sinusoidal external torque | 50.25 s |

The terminal gate checks lateral and axial port displacement, relative attitude, port-relative lateral velocity, and relative angular rate. The exact threshold values are implemented in the simulator's termination function.

## Analytic docking-port rate feed-forward

The target docking port moves as the target tumbles. The implementation analytically computes the target port elevation and azimuth rates and subtracts these rates from the LOS angular-rate tracking errors. This adds the moving-port rate to the tracking reference rather than treating the target motion as a disturbance that the reaching law must reject.

The source implementation is in `spacecraft_docking/frames.py` and `spacecraft_docking/control.py`.

## Online RLS-RFF disturbance observer

The observer maintains separate translational and rotational RLS regressors using 64 random Fourier features per channel. The predicted disturbance is subtracted from the equivalent control, and the observer-aided controller uses reduced reaching-law gains.

The comparison script runs each scenario with and without the observer and measures the RMS of consecutive-step differences in commanded thrust and torque. These are **command-chatter proxies**, not measurements of physical actuator vibration or propellant use.

| Scenario | Baseline docking time | Observer docking time | Thrust command variation reduction | Torque command variation reduction |
|---:|---:|---:|---:|---:|
| 1 | 46.55 s | 47.85 s | 53% | 32% |
| 2 | 46.70 s | 48.00 s | 56% | 20% |
| 3 | 48.20 s | 49.90 s | 40% | 16% |
| 4 | 50.25 s | 53.05 s | 56% | 9% |

All eight comparison runs (four baseline and four observer-aided) returned successful docking flags in the local verification run. The observer reduced the command-variation metrics while increasing docking time by approximately 1.3–2.8 seconds in these runs.

To reproduce the comparison:

```bash
python scripts/compare_observer.py
```

## Installation

Python dependencies are pinned in `requirements.txt`.

Create and activate a virtual environment, then install the dependencies:

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run simulations

Run one scenario:

```bash
python run.py --scenario 1
```

Run all four scenarios:

```bash
python run.py --scenario all
```

Run without sensor noise:

```bash
python run.py --scenario all --no-noise
```

Run with the RLS-RFF observer:

```bash
python run.py --scenario all --observer
```

Launch the interactive 3D viewer for one scenario:

```bash
python run.py --scenario 2 --interactive
```

Generate MP4 trajectory videos (requires `ffmpeg` available on PATH):

```bash
python run.py --scenario all --video
```

The simulator writes four figures and one CSV file per scenario under `outputs/`. MP4 files are written when `--video` is selected.

## Tests

The supplied test suite contains **50 regular tests** and **4 full-scenario integration tests** marked as slow. In the review of the supplied source archive, all 50 regular tests passed and all 4 slow integration tests passed.

Run the regular tests:

```bash
python -m pytest -q
```

Run the full-scenario integration tests:

```bash
python -m pytest -m slow -q
```

## Supplementary MATLAB EKF docking simulation

A separate MATLAB implementation is included in [`MATLAB/EKF_Docking_Simulation/`](MATLAB/EKF_Docking_Simulation/). It combines a seven-state EKF for target attitude quaternion and body angular velocity with CW relative translational dynamics, rigid-body attitude dynamics, and translational and attitude sliding-mode controllers.

The main entry point is `main_closedloop.m`; the EKF is implemented in `navigation_ekf.m`. The EKF uses a direct noisy measurement model for all seven state components (`H = I_7)). It does not estimate relative position and velocity, and it is not a complete camera/LiDAR navigation pipeline.

This MATLAB simulation is **separate from the Python nonlinear 6-DOF reproduction**. It uses CW linearised relative translation and torque-free target rotational propagation in the supplied main script. It does not implement the Python model's full disturbance set or the same five-condition terminal docking gate. Its plots should therefore not be conflated with the Python four-scenario docking-time and RLS-RFF results.

See the [MATLAB simulation README](MATLAB/EKF_Docking_Simulation/README.md) for run instructions and implementation limitations.

## Research reference

The baseline controller is reproduced from:

P. Abdollahzadeh and S. M. Esmailifar, “Automatic orbital docking with tumbling target using sliding mode control,” *Advances in Space Research*, vol. 67, no. 5, pp. 1506–1525, 2021.  
DOI: [10.1016/j.asr.2020.12.016](https://doi.org/10.1016/j.asr.2020.12.016).

This repository contains the implementation and generated simulation artefacts; it does not redistribute the publisher's paper PDF.

## Team and authorship

The course-project report supplied with the source archive lists the project team as:

- Sanjai V
- Satyadarshi Das
- Omkumar Dineshbhai Delvadia

This was documented as a team project. Individual technical contributions should be described accurately by the person presenting the work; this repository should not imply that all team contributions were performed by one person.

## Limitations

- The work is a numerical simulation, not a hardware or in-orbit demonstration.
- The model does not include discrete thruster allocation or plume-impingement effects.
- Atmospheric drag uses a simplified static exponential-atmosphere model.
- The supplementary MATLAB EKF uses an idealised direct-measurement assumption for the target quaternion and angular velocity; it is not a full relative-navigation sensor pipeline.
- Simulation success under the implemented model and terminal gate does not establish flight qualification or real spacecraft performance.

## Acknowledgement of LLM assistance

LLM-assisted tools were present in the supplied development workspace. The exact scope of their contribution to the code and documentation should be stated by the project authors based on how those tools were used. The project team remains responsible for reviewing and validating equations, implementation, test results, references, and all engineering claims.

## License

No open-source license is currently provided. Do not assume permission to reuse, modify, or redistribute this project's code or other materials.
