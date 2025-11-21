import gym
from gym import spaces
import numpy as np

from gym.envs.registration import register
from sympy.multipledispatch.dispatcher import RaiseNotImplementedError


class CircleWorldSingleIntegrator(gym.Env):
    """
    2D continuous circular world with *velocity* controls.

    State:  [x, y, vx, vy]
    Action: [vx_cmd, vy_cmd]  (desired velocity components)

    - Agent starts at (0, 0) with zero velocity every episode.
    - World is a circle of radius `radius`.
    - At each step, the action sets the velocity (clipped to max_speed).
    - Dynamics: x_{t+1} = x_t + vx * dt, y_{t+1} = y_t + vy * dt
    - Episode terminates if agent leaves the circle (optional) or max_steps reached.
    - Reward is 0 by default (pure dynamics sandbox).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        radius=1.0,
        dt=0.05,
        max_speed=2.0,
        max_steps=200,
        terminate_on_out_of_bounds=True, is_geometric_horizon=True, continuation_probability=0.99
    ):
        super(CircleWorldSingleIntegrator, self).__init__()

        self.radius = float(radius)
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.max_steps = int(max_steps)
        self.terminate_on_out_of_bounds = bool(terminate_on_out_of_bounds)

        # Observation: [x, y, vx, vy]
        low_obs = np.array(
            [-self.radius, -self.radius],
            dtype=np.float32,
        )
        high_obs = np.array(
            [self.radius, self.radius],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action: [vx_cmd, vy_cmd] (velocity components)
        self.action_space = spaces.Box(
            low=-self.max_speed,
            high=self.max_speed,
            shape=(2,),
            dtype=np.float32,
        )

        self.state = None  # [x, y]
        self.step_count = 0
        self.is_geometric_horizon = is_geometric_horizon
        self.continuation_probability = continuation_probability
        if self.is_geometric_horizon:
            self.max_steps = np.random.geometric(1.0-continuation_probability)

    # -------- Gym API --------

    def reset(self):
        """
        Classic Gym API: returns observation only.
        """
        if self.is_geometric_horizon:
            self.max_steps = np.random.geometric(1.0-self.continuation_probability)
        self.step_count = 0
        # Start at center with zero velocity
        self.state = np.zeros(2, dtype=np.float32)
        return self._get_obs()

    def _dynamics(self, state, action):
        """
        Continuous-time dynamics for single integrator.

        state: [x, y]
        action: [vx_cmd, vy_cmd]
        returns: [dx/dt, dy/dt]
        """
        x, y = state
        vx_cmd, vy_cmd = action
        dx = vx_cmd
        dy = vy_cmd
        return np.array([dx, dy], dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # Clip commanded velocity to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        vx_cmd, vy_cmd = action

        s = self.state.astype(np.float32)

        # ----- RK4 integration for single integrator -----
        # s' = f(s, a), with a = constant over [t, t+dt]
        k1 = self._dynamics(s, action)
        k2 = self._dynamics(s + 0.5 * self.dt * k1, action)
        k3 = self._dynamics(s + 0.5 * self.dt * k2, action)
        k4 = self._dynamics(s + self.dt * k3, action)

        s_next = s + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x, y = s_next
        self.state = np.array([x, y], dtype=np.float32)
        self.step_count += 1

        # Check boundary
        r = np.sqrt(x * x + y * y)
        out_of_bounds = r > self.radius + 1e-6

        done = False
        if self.terminate_on_out_of_bounds and out_of_bounds:
            done = True
        if self.step_count >= self.max_steps:
            done = True

        reward = 0.0  # sandbox

        info = {
            "radius": float(r),
            "out_of_bounds": bool(out_of_bounds),
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.clip(self.state, self.observation_space.low, self.observation_space.high)

    def render(self, mode="human"):
        # You can reuse the matplotlib/pygame render logic from the other env
        pass

    def close(self):
        pass


# -------- Registration with Gym --------

try:
    register(
        id="Circle-World-Single-Integrator-v0",
        entry_point="gym_extensions.continuous.circle_world_single_integrator:CircleWorldSingleIntegrator",
        max_episode_steps=200,
    )
except gym.error.Error:
    # Already registered
    pass
