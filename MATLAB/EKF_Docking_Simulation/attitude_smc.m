function torque_cmd = attitude_smc(q_chaser, w_chaser, q_target, w_target, I_chaser)
% Sliding-mode attitude synchronisation using a relative quaternion.

    lambda = 0.5;
    K_robust = 5.0;

    % Relative quaternion q_error = conjugate(q_target) * q_chaser.
    q_t_inv = [q_target(1); -q_target(2:4)];
    Q_mat = [ q_t_inv(1), -q_t_inv(2), -q_t_inv(3), -q_t_inv(4);
              q_t_inv(2),  q_t_inv(1), -q_t_inv(4),  q_t_inv(3);
              q_t_inv(3),  q_t_inv(4),  q_t_inv(1), -q_t_inv(2);
              q_t_inv(4), -q_t_inv(3),  q_t_inv(2),  q_t_inv(1)];
    q_error = Q_mat * q_chaser;

    q_ev = q_error(2:4);
    w_error = w_chaser - w_target;

    S = w_error + lambda * q_ev;

    q_ev_dot = 0.5 * (q_error(1) * w_error + ...
                      cross(q_ev, w_chaser + w_target));

    tau_nominal = cross(w_chaser, I_chaser * w_chaser) ...
                  - I_chaser * (lambda * q_ev_dot);
    tau_robust = -K_robust * sign(S);

    torque_cmd = tau_nominal + tau_robust;
end
