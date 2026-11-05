% Closed-Loop Docking Simulation (CW Translation + Attitude SMC + EKF)
% Generates trajectory/attitude/EKF plots and an MP4 animation.

clear; clc; close all;

%% 1. Simulation Parameters
dt = 0.1;
t_final = 300;
time = 0:dt:t_final;
N = length(time);

%% 2. Orbital & Spacecraft Setup
mu = 3.986e14;
R_earth = 6371e3;
alt = 400e3;
a = R_earth + alt;
n = sqrt(mu / a^3);

mass_chaser = 500;
I_chaser = diag([250, 300, 350]);

mass_target = 400; %#ok<NASGU> % Retained from the original setup; not used by this model.
I_target = diag([200, 250, 300]);

%% 3. Initialize States
chaser_cw_state = zeros(6, N);
chaser_cw_state(:, 1) = [50; -100; 25; 0; 0; 0];

target_rot_state = zeros(7, N);
target_rot_state(:, 1) = [1; 0; 0; 0; 0.1; 0.05; -0.02];

chaser_rot_state = zeros(7, N);
chaser_rot_state(:, 1) = [1; 0; 0; 0; 0; 0; 0];

%% 4. Navigation Setup (Extended Kalman Filter)
sensor_noise_quat = 0.05;
sensor_noise_omega = 0.02;

R_kf = diag([repmat(sensor_noise_quat^2, 1, 4), ...
             repmat(sensor_noise_omega^2, 1, 3)]);
Q_kf = eye(7) * 1e-6;
P_kf = eye(7) * 0.1;

target_est_state = zeros(7, N);
target_est_state(:, 1) = [1; 0; 0; 0; 0; 0; 0];

%% 5. Closed-Loop RK4 Integration
disp('Running closed-loop docking simulation (SMC + EKF)...');

for k = 1:(N-1)
    t = time(k);

    % A. Propagate target rotational dynamics
    current_target_rot = target_rot_state(:, k);
    next_target_rot = rk4_step(@attitude_dynamics, t, current_target_rot, ...
                               dt, I_target, [0; 0; 0]);
    next_target_rot(1:4) = next_target_rot(1:4) / norm(next_target_rot(1:4));
    target_rot_state(:, k+1) = next_target_rot;

    % A.2 Generate noisy target attitude/rate measurement and update EKF
    noise_vector = [randn(4, 1) * sensor_noise_quat; ...
                    randn(3, 1) * sensor_noise_omega];
    z_meas = current_target_rot + noise_vector;

    prev_est = target_est_state(:, k);
    [new_est, P_kf] = navigation_ekf(prev_est, P_kf, z_meas, ...
                                     dt, I_target, Q_kf, R_kf);
    target_est_state(:, k+1) = new_est;

    % B. Chaser guidance from estimated target docking-port state
    q_t_est = new_est(1:4);
    w_t_est = new_est(5:7);

    r_port_body = [2; 0; 0];
    q0 = q_t_est(1); q1 = q_t_est(2);
    q2 = q_t_est(3); q3 = q_t_est(4);

    R_matrix = [q0^2+q1^2-q2^2-q3^2, 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2);
                2*(q1*q2+q0*q3),   q0^2-q1^2+q2^2-q3^2, 2*(q2*q3-q0*q1);
                2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   q0^2-q1^2-q2^2+q3^2];

    r_port_lvlh = R_matrix * r_port_body;
    v_port_lvlh = cross(w_t_est, r_port_lvlh);
    target_pos_vel_est = [r_port_lvlh; v_port_lvlh];

    force_cmd = guidance_smc(chaser_cw_state(:, k), ...
                             target_pos_vel_est, mass_chaser, n);

    % B.2 Chaser attitude synchronization
    q_c = chaser_rot_state(1:4, k);
    w_c = chaser_rot_state(5:7, k);
    torque_cmd = attitude_smc(q_c, w_c, q_t_est, w_t_est, I_chaser);

    % C. Propagate chaser translational and rotational dynamics
    current_chaser_cw = chaser_cw_state(:, k);
    next_chaser_cw = rk4_step(@cw_dynamics, t, current_chaser_cw, ...
                              dt, n, mass_chaser, force_cmd);
    chaser_cw_state(:, k+1) = next_chaser_cw;

    current_chaser_rot = chaser_rot_state(:, k);
    next_chaser_rot = rk4_step(@attitude_dynamics, t, current_chaser_rot, ...
                               dt, I_chaser, torque_cmd);
    next_chaser_rot(1:4) = next_chaser_rot(1:4) / norm(next_chaser_rot(1:4));
    chaser_rot_state(:, k+1) = next_chaser_rot;
end

disp('Simulation complete. Generating plots and animation...');

%% 6. Results plots
fig1 = figure('Name', '3D SMC Docking Trajectory', ...
              'Position', [100, 100, 700, 500]);
plot3(chaser_cw_state(2, :), chaser_cw_state(1, :), ...
      chaser_cw_state(3, :), 'b-', 'LineWidth', 2);
hold on;
plot3(chaser_cw_state(2, 1), chaser_cw_state(1, 1), ...
      chaser_cw_state(3, 1), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
plot3(0, 0, 0, 'rx', 'MarkerSize', 12, 'LineWidth', 3);
xlabel('Along-Track y (m)', 'FontWeight', 'bold');
ylabel('Radial x (m)', 'FontWeight', 'bold');
zlabel('Cross-Track z (m)', 'FontWeight', 'bold');
title('3D Relative Chaser Trajectory via SMC & EKF');
grid on; view(3);
legend('Chaser Path', 'Start Point', 'Target Spacecraft', 'Location', 'best');
saveas(fig1, 'Result_1_Trajectory.png');

fig2 = figure('Name', 'Attitude Synchronization', ...
              'Position', [850, 100, 600, 400]);
plot(time, target_rot_state(5:7, :), '--', 'LineWidth', 1.5);
hold on;
plot(time, chaser_rot_state(5:7, :), '-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('\omega (rad/s)');
title('Chaser Synchronizing with Target Tumble');
grid on;
legend('\omega_{xt}', '\omega_{yt}', '\omega_{zt}', ...
       '\omega_{xc}', '\omega_{yc}', '\omega_{zc}');
saveas(fig2, 'Result_2_Attitude.png');

fig3 = figure('Name', 'EKF Performance', ...
              'Position', [450, 600, 600, 400]);
plot(time, target_rot_state(5, :), 'k-', 'LineWidth', 2);
hold on;
% This display noise is generated separately for visualisation. It is not the
% exact noisy measurement sequence used internally by the EKF.
plot(time, target_rot_state(5, :) + randn(1, N) * sensor_noise_omega, ...
     'r.', 'MarkerSize', 1);
plot(time, target_est_state(5, :), 'b--', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('\omega_x (rad/s)');
title('EKF Sensor Fusion: True vs Illustrative Noisy vs Estimated');
legend('True Tumble', 'Illustrative Noisy Signal', 'EKF Estimate');
saveas(fig3, 'Result_3_EKF.png');

%% 7. 3D animation and MP4 export
fig4 = figure('Name', '6-DOF Docking Animation', ...
              'Position', [100, 100, 800, 600]);

video_filename = 'Simulation_Animation.mp4';
v = VideoWriter(video_filename, 'MPEG-4');
v.FrameRate = 30;
open(v);

skip_frames = 10;
for k = 1:skip_frames:N
    clf;
    hold on; grid on; view(3);

    xlim([-20, 60]); ylim([-120, 20]); zlim([-20, 40]);
    xlabel('Radial x (m)', 'FontWeight', 'bold');
    ylabel('Along-Track y (m)', 'FontWeight', 'bold');
    zlabel('Cross-Track z (m)', 'FontWeight', 'bold');
    title(sprintf('Chaser Docking Animation | Time: %.1f s', time(k)));

    q_t = target_rot_state(1:4, k);
    R_t = quat_to_dcm(q_t);

    X_t = R_t * [4; 0; 0];
    Y_t = R_t * [0; 4; 0];
    Z_t = R_t * [0; 0; 4];

    plot3(0, 0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    plot3([0, X_t(1)], [0, X_t(2)], [0, X_t(3)], 'r-', 'LineWidth', 4);
    plot3([0, Y_t(1)], [0, Y_t(2)], [0, Y_t(3)], 'g-', 'LineWidth', 2);
    plot3([0, Z_t(1)], [0, Z_t(2)], [0, Z_t(3)], 'b-', 'LineWidth', 2);

    pos_c = chaser_cw_state(1:3, k);
    q_c = chaser_rot_state(1:4, k);
    R_c = quat_to_dcm(q_c);

    X_c = pos_c + R_c * [4; 0; 0];
    Y_c = pos_c + R_c * [0; 4; 0];
    Z_c = pos_c + R_c * [0; 0; 4];

    plot3(pos_c(1), pos_c(2), pos_c(3), ...
          'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    plot3([pos_c(1), X_c(1)], [pos_c(2), X_c(2)], ...
          [pos_c(3), X_c(3)], 'r-', 'LineWidth', 4);
    plot3([pos_c(1), Y_c(1)], [pos_c(2), Y_c(2)], ...
          [pos_c(3), Y_c(3)], 'g-', 'LineWidth', 2);
    plot3([pos_c(1), Z_c(1)], [pos_c(2), Z_c(2)], ...
          [pos_c(3), Z_c(3)], 'b-', 'LineWidth', 2);

    plot3(chaser_cw_state(1, 1:k), chaser_cw_state(2, 1:k), ...
          chaser_cw_state(3, 1:k), 'k:', 'LineWidth', 1);

    drawnow;
    frame = getframe(fig4);
    writeVideo(v, frame);
end

close(v);
disp(['Plots and video saved to: ', pwd]);

%% Local helper function
function R = quat_to_dcm(q)
    q0 = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
    R = [q0^2+q1^2-q2^2-q3^2, 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2);
         2*(q1*q2+q0*q3),   q0^2-q1^2+q2^2-q3^2, 2*(q2*q3-q0*q1);
         2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   q0^2-q1^2-q2^2+q3^2];
end
