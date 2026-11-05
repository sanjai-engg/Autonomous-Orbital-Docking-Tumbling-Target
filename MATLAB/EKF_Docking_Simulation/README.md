# MATLAB EKF-Assisted Tumbling-Target Docking Simulation

## Purpose

This folder contains a MATLAB closed-loop simulation that tests an estimator-in-the-loop docking architecture: the target's attitude quaternion and angular velocity are estimated by a seven-state Extended Kalman Filter (EKF), and those estimates are passed to translational and attitude sliding-mode controllers (SMCs).

The implementation is supplementary to the Python simulation in the parent repository. It is not the same nonlinear six-degree-of-freedom model, does not reproduce the same four Python benchmark scenarios, and does not use the Python RLS-RFF disturbance observer.

## Closed-loop sequence

At each simulation step, `main_closedloop.m` performs the following operations:

1. **Target rotational propagation:** propagate the target's quaternion and body angular velocity using rigid-body Euler equations and quaternion kinematics. The target torque input in this script is zero, so the target undergoes torque-free tumble.
2. **Synthetic measurement generation:** add Gaussian noise to all four quaternion components and all three angular-velocity components of the target state.
3. **EKF update:** call `navigation_ekf.m` to predict the target state and covariance, form a numerical state-transition Jacobian, calculate the Kalman gain, correct the seven-state estimate, and renormalise the quaternion.
4. **Docking-port estimate:** rotate the fixed body-frame port offset `[2; 0; 0]` using the estimated target quaternion and calculate an estimated port velocity from angular velocity and port position.
5. **Translational guidance:** call `guidance_smc.m`. Its sliding surface combines position and velocity tracking errors; its nominal term cancels the CW orbital acceleration and its robust term drives the surface toward zero.
6. **Attitude guidance:** call `attitude_smc.m`. It uses the relative quaternion vector part and angular-velocity error, with nominal gyroscopic compensation and a sign-based robust torque.
7. **Chaser propagation:** integrate chaser CW translation and chaser rigid-body attitude using the RK4 helper.

The simulation runs with a 0.1 s integration step for 300 s. It saves a relative 3D trajectory plot, an attitude-rate plot, an EKF estimate plot, and an MP4 animation in the MATLAB current working directory.

## EKF state, model, and assumptions

The EKF state is

`x = [q0; q1; q2; q3; wx; wy; wz]`

where the quaternion is scalar-first. The measurement vector is generated as the true seven-state target vector plus independent Gaussian noise. The observation matrix is `H = eye(7)`, so the filter assumes direct noisy measurements of every state component.

The filter prediction uses `attitude_dynamics.m` and `rk4_step.m`. Its state-transition Jacobian is approximated by forward finite differences with a perturbation size of `1e-5`. The covariance is propagated with the supplied process-noise matrix, then corrected with the Kalman gain. The quaternion is normalised after the additive state correction.

This is a simplified simulated sensor-fusion model. It is **not** a camera/LiDAR measurement pipeline and does not estimate relative position or velocity.

## Controller and dynamics files

| File | Function |
|---|---|
| `main_closedloop.m` | Runs target propagation, synthetic measurement generation, EKF, SMC guidance, chaser propagation, plots, and animation |
| `navigation_ekf.m` | Seven-state target attitude/angular-rate EKF |
| `attitude_dynamics.m` | Quaternion kinematics and Euler rigid-body rotational dynamics |
| `cw_dynamics.m` | CW relative translational dynamics with applied force |
| `rk4_step.m` | Fourth-order Runge–Kutta integration step |
| `guidance_smc.m` | Translational SMC using CW nominal cancellation and a robust reaching term |
| `attitude_smc.m` | Quaternion-based attitude SMC |

## Run

1. Open MATLAB.
2. Set the MATLAB Current Folder to this directory, or add this directory to the MATLAB path.
3. Run:

```matlab
main_closedloop
```

The script uses MATLAB's `VideoWriter` with the `MPEG-4` profile to create the animation. MP4 export depends on the installed MATLAB release and platform support.

## Important limitations and checks

- **Direct-state measurement assumption:** the EKF is given noisy quaternion and angular-rate values directly. This assumes all seven states are observable and bypasses camera/LiDAR feature extraction and pose reconstruction.
- **Quaternion update:** the filter uses an additive correction to quaternion components followed by normalisation; this is not a multiplicative/error-state quaternion EKF.
- **Jacobian perturbation:** the perturbed quaternion is not normalised before each finite-difference propagation. This should be considered when interpreting the numerical Jacobian.
- **Covariance update:** the implementation uses `(I-KH)P`, not the Joseph stabilised covariance update.
- **Frame consistency of port velocity:** `main_closedloop.m` rotates the port vector into the simulation frame and then computes its cross product with the estimated angular velocity. Verify that angular velocity is expressed in the same frame under the chosen quaternion convention; transform it if it is still expressed in target body coordinates.
- **No docking terminal gate:** this script does not evaluate the five-condition terminal gate used by the Python simulator and does not output a quantified docking-success time.
- **Different translational model:** the MATLAB implementation uses linear CW equations; it does not include the Python model's nonlinear gravity, atmospheric drag, differential (J_2), or gravity-gradient disturbance terms.
- **Plot interpretation:** the noisy angular-rate trace in the EKF performance plot is generated separately for visualisation and is not the exact noisy measurement sequence consumed by the EKF.
- **Execution status:** the MATLAB source has been reviewed, but the script has not been independently executed in MATLAB during this repository review.

These points define the limits of this implementation. The code is a useful estimator/controller integration study, but its results should not be described as flight-ready navigation performance.

## Research basis

The broader project reproduces and extends the sliding-mode docking controller described in:

P. Abdollahzadeh and S. M. Esmailifar, “Automatic orbital docking with tumbling target using sliding mode control,” *Advances in Space Research*, vol. 67, no. 5, pp. 1506–1525, 2021.  
DOI: [10.1016/j.asr.2020.12.016](https://doi.org/10.1016/j.asr.2020.12.016).
