# MATLAB EKF Docking Simulation

This folder contains a MATLAB closed-loop docking simulation that combines:

- A seven-state Extended Kalman Filter (EKF) for target attitude-quaternion and body-angular-velocity estimation.
- Clohessy–Wiltshire (CW) relative translational dynamics for the chaser.
- Rigid-body rotational dynamics for the chaser and target.
- Translational and attitude sliding-mode controllers (SMC).
- RK4 time integration, plots, and an MP4 animation.

This MATLAB implementation is a separate supplementary simulation. It should not be treated as identical to the Python nonlinear 6-DOF reproduction or as an independent reproduction of all four published-paper scenarios.

## Files

| File | Role |
|---|---|
| `main_closedloop.m` | Main closed-loop simulation, EKF integration, plotting, and animation export |
| `navigation_ekf.m` | Seven-state EKF prediction and correction |
| `attitude_dynamics.m` | Quaternion kinematics and Euler rigid-body rotational equations |
| `cw_dynamics.m` | CW relative translational equations with commanded force |
| `rk4_step.m` | One fourth-order Runge–Kutta integration step |
| `guidance_smc.m` | Translational sliding-mode guidance |
| `attitude_smc.m` | Quaternion-based attitude sliding-mode control |

## Run

1. Open MATLAB.
2. Set the MATLAB Current Folder to this directory, or add this directory to the MATLAB path.
3. Run:

```matlab
main_closedloop
```

The simulation uses a 0.1 s integration step over 300 s. It generates three PNG plots and an MP4 animation in the current working directory. The animation uses MATLAB's `VideoWriter` with the `MPEG-4` profile; availability depends on the MATLAB installation and platform.

## EKF state and measurement assumptions

The EKF state is

`[q0; q1; q2; q3; wx; wy; wz]`

with a scalar-first quaternion. The current simulation adds Gaussian noise directly to all four quaternion components and all three angular-velocity components. The measurement matrix is set to the identity, `H = eye(7)`; therefore the implementation assumes that all seven state components are directly observed with noise. This is an idealised simulated measurement model, not a complete camera/LiDAR measurement-processing pipeline.

The filter predicts the target state using the rigid-body attitude model and RK4, approximates the state-transition Jacobian numerically, performs a Kalman correction, and normalises the quaternion after correction.

## Model scope and limitations

- Translational chaser motion uses the linear CW equations in the LVLH frame.
- The target's rotational state is propagated using torque-free rigid-body dynamics in this script.
- The target docking-port position is obtained by rotating a fixed body-frame offset; its estimated velocity is computed from the estimated angular velocity.
- The current script does not include the Python model's nonlinear differential-gravity, atmospheric-drag, (J_2), and gravity-gradient disturbance implementation.
- The EKF applies an additive correction to quaternion components followed by normalisation. It is not a multiplicative/error-state quaternion EKF.
- The numerical Jacobian perturbs state components directly; the perturbed quaternion is not normalised before each perturbed propagation. This is a numerical-consistency limitation to address in a future filter refinement.
- The covariance update uses the simplified `(I-KH)P` expression rather than the Joseph stabilised form.
- The EKF uses direct noisy measurements of the full seven-state vector. It does not estimate relative position and velocity.
- The script does not implement the same five-condition terminal docking gate used by the Python scenario simulator. Its plots and animation should not be represented as a quantified pass of those Python terminal criteria.

These limitations are stated to define the scope of this implementation, not to claim flight-ready navigation performance.

## Relationship to the Python simulation

The Python package contains the main paper-controller reproduction, its four scenario runs, and the RLS-RFF disturbance-observer comparison. This MATLAB folder is a distinct EKF-assisted closed-loop simulation using CW translation and attitude SMC. Keep numerical claims and results associated with each implementation clearly labelled.

## Research basis

The original sliding-mode docking controller reproduced in the broader project is:

P. Abdollahzadeh and S. M. Esmailifar, “Automatic orbital docking with tumbling target using sliding mode control,” *Advances in Space Research*, vol. 67, no. 5, pp. 1506–1525, 2021.  
DOI: [10.1016/j.asr.2020.12.016](https://doi.org/10.1016/j.asr.2020.12.016).
