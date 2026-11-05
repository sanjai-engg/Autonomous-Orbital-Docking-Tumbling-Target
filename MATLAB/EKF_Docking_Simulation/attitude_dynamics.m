function state_dot = attitude_dynamics(~, state, inertia_tensor, control_torque)
% Rigid-body rotational dynamics and scalar-first quaternion kinematics.
% State: [q0; q1; q2; q3; wx; wy; wz].

    q = state(1:4);
    omega = state(5:7);

    wx = omega(1);
    wy = omega(2);
    wz = omega(3);

    % Euler rigid-body equation:
    % I*omega_dot + omega x (I*omega) = applied torque.
    omega_dot = inertia_tensor \ ...
        (control_torque - cross(omega, inertia_tensor * omega));

    omega_matrix = [
         0, -wx, -wy, -wz;
        wx,   0,  wz, -wy;
        wy, -wz,   0,  wx;
        wz,  wy, -wx,   0
    ];

    q_dot = 0.5 * omega_matrix * q;
    state_dot = [q_dot; omega_dot];
end
