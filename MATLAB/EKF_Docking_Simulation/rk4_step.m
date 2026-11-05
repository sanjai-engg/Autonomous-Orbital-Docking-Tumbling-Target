function state_next = rk4_step(dynamics_func, t, state, dt, varargin)
% One fourth-order Runge-Kutta integration step.

    k1 = dynamics_func(t, state, varargin{:});
    k2 = dynamics_func(t + dt/2, state + (dt/2)*k1, varargin{:});
    k3 = dynamics_func(t + dt/2, state + (dt/2)*k2, varargin{:});
    k4 = dynamics_func(t + dt, state + dt*k3, varargin{:});

    state_next = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
end
