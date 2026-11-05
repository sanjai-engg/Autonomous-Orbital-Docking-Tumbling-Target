function force_cmd = guidance_smc(state_chaser, state_target_port, mass_chaser, n)
% Sliding-mode translational guidance for tracking a moving docking port.

    c_slope = 0.05;
    K_robust = 0.5;  % Acceleration gain (m/s^2).

    pos_chaser = state_chaser(1:3);
    vel_chaser = state_chaser(4:6);

    pos_target = state_target_port(1:3);
    vel_target = state_target_port(4:6);

    pos_error = pos_chaser - pos_target;
    vel_error = vel_chaser - vel_target;

    % Sliding surface.
    S = vel_error + c_slope * pos_error;

    % Nominal acceleration cancelling the CW orbital terms.
    x = pos_chaser(1);
    z = pos_chaser(3);
    vy = vel_chaser(5-3); % y-velocity is the fifth element of the state.
    vx = vel_chaser(1);
    a_cw = [3*n^2*x + 2*n*vy;
            -2*n*vx;
            -n^2*z];

    a_nominal = -a_cw - c_slope * vel_error;
    a_robust = -K_robust * sign(S);

    force_cmd = mass_chaser * (a_nominal + a_robust);
end
