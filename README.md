# Autonomous Orbital Docking with a Tumbling Target

A Python-based six-degree-of-freedom (6-DOF) simulation study reproducing the sliding-mode controller proposed by P. Abdollahzadeh and S. M. Esmailifar for autonomous docking with a freely tumbling target spacecraft.

The project reproduces four published docking scenarios and studies implementation extensions for target docking-port motion compensation and online disturbance estimation. A separate MATLAB extended Kalman filter (EKF) sensor-fusion extension is also documented in the project report.

> **Scope:** This is a numerical simulation and controller-reproduction study. It is not a hardware demonstration or an in-orbit docking test.

## Project highlights

- **6-DOF closed-loop simulation:** Relative translational dynamics in the local-vertical/local-horizontal (LVLH) frame, quaternion attitude kinematics, and rotational dynamics for chaser and target.
- **Sliding-mode control:** Translational guidance in line-of-sight (LOS) coordinates and rotational control using the relative quaternion.
- **Environmental disturbances and sensor noise:** Atmospheric drag, Earth's (J_2) perturbation, gravity-gradient torque, and a Gaussian measurement-noise model.
- **Four docking scenarios:** The four scenarios reported in the reproduction report satisfy the stated Appendix-D terminal criteria, with docking times from **46.55 s to 50.25 s**.
- **Target-motion feed-forward:** Analytic docking-port angular-rate compensation is included in the LOS tracking surface to address the one-step target-reference lag in the zero-order-hold (ZOH) implementation.
- **Online RLS-RFF disturbance observer:** An online Recursive Least Squares / Random Fourier Features observer is integrated with the SMC. The report measures **40–56% reduction in thrust-command chatter** and **9–32% reduction in torque-command chatter** across the four scenarios.
- **MATLAB EKF extension:** A separate 7-state EKF for target attitude and angular-rate estimation is described in the report.

## Repository structure

The following structure is intended for the source code, tests, and generated outputs. The course report PDF is intentionally not included; the repository is focused on the implementation and reproducible technical artefacts:

```text
Autonomous-Orbital-Docking-Tumbling-Target/
├── README.md
├── Python/
│   ├── frames.py
│   ├── dynamics.py
│   ├── disturbances.py
│   ├── control.py
│   ├── sensors.py
│   ├── scenarios.py
│   ├── simulator.py
│   ├── plotting.py
│   └── viewer.py
├── MATLAB/
│   └── EKF_Sensor_Fusion/
├── Tests/
├── Results/
│   ├── Figures/
│   └── Trajectories/
├── requirements.txt
└── .gitignore
```

The module names above follow the software architecture documented in the reproduction report. Add the actual source files and test suite before treating this layout as a complete implementation.

## Numerical model and control architecture

The simulation represents the chaser and target states using relative position and velocity, chaser and target attitude quaternions, and body angular velocities. Translational motion is expressed in the target-centred LVLH frame. The translational SMC uses LOS range, elevation, and azimuth errors; the rotational SMC uses the relative quaternion vector part and relative angular velocity.

The closed loop uses a **50 ms control period (20 Hz)** and integrates the plant over each ZOH interval using SciPy's RK45 solver. Chaser and target quaternions are renormalised after integration.

## Reported docking results

| Scenario | Target condition | Docking time | Result |
|---:|---|---:|---|
| 1 | No initial tumble; no external torque | 46.55 s | Meets terminal criteria |
| 2 | Initial target angular velocity | 46.70 s | Meets terminal criteria |
| 3 | Sinusoidal external torque | 48.20 s | Meets terminal criteria |
| 4 | Initial tumble and sinusoidal external torque | 50.25 s | Meets terminal criteria |

The terminal gate checks the report's five Appendix-D conditions simultaneously: lateral and axial port displacement, relative attitude, lateral velocity, and relative angular rate. Refer to the full report for the precise thresholds and scenario definitions.

## Implementation extensions

### Analytic docking-port angular-rate feed-forward

The target's docking-port direction changes as the target tumbles. The implementation analytically computes the port-angle rates and subtracts them from the LOS angular-rate tracking errors. The report analyses this as a correction to the one-step lag in the ZOH loop. It estimates the remaining higher-order residual at approximately **0.001° per control step** in the stated worst-case estimate.

### Online RLS-RFF disturbance observer

Two RLS-RFF observer channels estimate translational and rotational disturbance terms. The predicted disturbance is subtracted from the equivalent control before the SMC reaching term is applied.

| Scenario | Thrust chatter reduction | Torque chatter reduction |
|---:|---:|---:|
| 1 | 40% | 9% |
| 2 | 45% | 15% |
| 3 | 52% | 28% |
| 4 | 56% | 32% |

The report also records a docking-time increase of approximately 1–2 seconds in the observer runs relative to the corresponding baseline cases.

### MATLAB EKF sensor-fusion extension

The report separately describes a 7-state EKF estimating target attitude quaternion and body angular velocity from noisy simulated measurements. Keep this extension distinct from the baseline Python SMC reproduction when organising and describing the code.

## Tests

The report describes **48 automated Python tests**: 44 unit tests and four slower end-to-end integration tests, one for each published docking scenario. The actual test files should be included in `Tests/` so these claims can be independently inspected and run.

## Software and dependencies

The simulation report identifies Python modules using NumPy, SciPy, and plotting/visualisation tools, and describes the use of pytest for automated tests. A reproducible `requirements.txt` should be prepared from the imports and versions used by the actual source code.

Once the source files and dependency list are added, document the tested Python version and the commands for running unit tests, integration tests, and scenario simulations.

## Research basis

This project reproduces and extends the controller described in:

P. Abdollahzadeh and S. M. Esmailifar, “Automatic orbital docking with tumbling target using sliding mode control,” *Advances in Space Research*, vol. 67, no. 5, pp. 1506–1525, 2021.  
DOI: [10.1016/j.asr.2020.12.016](https://doi.org/10.1016/j.asr.2020.12.016).

The report also cites the Random Fourier Features method of Rahimi and Recht (2007), as well as standard references for astrodynamics, spacecraft attitude dynamics, and sliding-mode control.

## Team and contributions

The reproduction report lists the following project team:

- Sanjai V
- Satyadarshi Das
- Omkumar Dineshbhai Delvadia

The report documents the overall implementation study and its extensions. Individual contributions should be described separately where applicable; this repository should not imply that all team work was performed independently by one contributor.

## Limitations

The report identifies modelling limitations including the absence of thruster allocation and plume-impingement effects, use of a static exponential atmospheric model, and assumptions in the EKF sensor-observability model. This work is a simulation study; it does not establish flight qualification or hardware readiness.

## Acknowledgement of LLM assistance

ChatGPT was used as a supporting tool to help draft and organise this repository README and its project structure. This assistance did not include implementing or independently validating the simulation code, deriving the reported numerical results, or performing the project experiments. The project authors remain responsible for the technical content, equations, code, results, references, and all engineering claims.

## License

No open-source license is currently provided. Do not assume permission to reuse, modify, or redistribute the source code, report, or other project materials.
