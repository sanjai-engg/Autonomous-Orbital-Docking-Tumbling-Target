function state_dot = cw_dynamics(~, state, n, mass, control_force)
% Clohessy-Wiltshire relative translational dynamics with applied force.
% State: [x; y; z; vx; vy; vz], expressed in the LVLH frame.

    x = state(1);
    y = state(2);
    z = state(3);
    vx = state(4);
    vy = state(5);
    vz = state(6);

    Fx = control_force(1);
    Fy = control_force(2);
    Fz = control_force(3);

    ax = 3 * n^2 * x + 2 * n * vy + Fx / mass;
    ay = -2 * n * vx + Fy / mass;
    az = -n^2 * z + Fz / mass;

    state_dot = [vx; vy; vz; ax; ay; az];
end
