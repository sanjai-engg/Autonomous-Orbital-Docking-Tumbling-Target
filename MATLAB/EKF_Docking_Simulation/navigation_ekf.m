function [x_est_new, P_new] = navigation_ekf(x_est_prev, P_prev, z_meas, dt, I_target, Q_kf, R_kf)
% 7-state EKF for tumbling-target attitude and angular-rate estimation.
% State: [q0; q1; q2; q3; wx; wy; wz] (scalar-first quaternion).

    n_states = 7;

    % Prediction using the rigid-body attitude model and RK4.
    x_pred = rk4_step(@attitude_dynamics, 0, x_est_prev, dt, I_target, [0; 0; 0]);
    x_pred(1:4) = x_pred(1:4) / norm(x_pred(1:4));

    % Numerically approximate the state-transition Jacobian.
    F = eye(n_states);
    epsilon = 1e-5;
    for i = 1:n_states
        x_perturb = x_est_prev;
        x_perturb(i) = x_perturb(i) + epsilon;
        x_next_perturb = rk4_step(@attitude_dynamics, 0, x_perturb, ...
                                  dt, I_target, [0; 0; 0]);
        F(:, i) = (x_next_perturb - x_pred) / epsilon;
    end

    % Covariance prediction.
    P_pred = F * P_prev * F' + Q_kf;

    % Direct noisy measurement model for all seven state components.
    H = eye(n_states);
    S = H * P_pred * H' + R_kf;
    K = P_pred * H' / S;

    % Measurement innovation and state correction.
    y_residual = z_meas - H * x_pred;
    x_est_new = x_pred + K * y_residual;
    x_est_new(1:4) = x_est_new(1:4) / norm(x_est_new(1:4));

    % Covariance update as implemented in the supplied code.
    P_new = (eye(n_states) - K * H) * P_pred;
end
