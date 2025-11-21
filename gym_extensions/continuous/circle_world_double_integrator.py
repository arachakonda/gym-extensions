import gym
from gym import spaces
import numpy as np

# Optional: if you want to auto-register on import
from gym.envs.registration import register

from matplotlib import pyplot as plt


class CircleWorldDoubleIntegrator(gym.Env):
    """
    2D continuous circular world with second-order dynamics.

    State:  [x, y, vx, vy]
    Action: [ax, ay] (accelerations)

    - Agent starts at (0, 0) with zero velocity every episode.
    - World is a circle of radius `radius`.
    - Terminate when agent leaves the circle (optional) or max_steps is reached.
    - Reward is 0 by default (sandbox dynamics); plug in your own reward.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        radius=1.0,
        dt=0.05,
        max_speed=2.0,
        max_accel=1.0,
        max_steps=200,
        terminate_on_out_of_bounds=True,
        render_mode = None,
        is_geometric_horizon=True,
        continuation_probability=0.99
    ):
        super(CircleWorldDoubleIntegrator, self).__init__()

        self.radius = float(radius)
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.max_steps = int(max_steps)
        self.terminate_on_out_of_bounds = bool(terminate_on_out_of_bounds)

        # Observation: [x, y, vx, vy]
        low_obs = np.array(
            [-self.radius, -self.radius, -self.max_speed, -self.max_speed],
            dtype=np.float32,
        )
        high_obs = np.array(
            [self.radius, self.radius, self.max_speed, self.max_speed],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action: [ax, ay] acceleration
        self.action_space = spaces.Box(
            low=-self.max_accel,
            high=self.max_accel,
            shape=(2,),
            dtype=np.float32,
        )

        # Internal state: [x, y, vx, vy]
        self.state = None
        self.step_count = 0
        self.continuation_probability = continuation_probability
        self.is_geometric_horizon = is_geometric_horizon
        if self.is_geometric_horizon:
            self.max_steps = np.random.geometric(1.0-self.continuation_probability)

        self.render_mode = render_mode

    # ------------- Gym API -------------

    def reset(self):
        """
        Gym-extensions uses the older Gym API (no seed, no info).
        """
        if self.is_geometric_horizon:
            self.max_steps = np.random.geometric(1.0-self.continuation_probability)
        self.step_count = 0
        # Start at center, zero velocity
        self.state = np.zeros(4, dtype=np.float32)
        return self._get_obs()

    def _dynamics(self, state, action):
        """
        Continuous-time dynamics for the point mass.

        state = [x, y, vx, vy]
        action = [ax, ay]
        """
        x, y, vx, vy = state
        ax, ay = action
        dx = vx
        dy = vy
        dvx = ax
        dvy = ay
        return np.array([dx, dy, dvx, dvy], dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # Clip action to [-max_accel, max_accel]
        action = np.clip(action, self.action_space.low, self.action_space.high)

        s = self.state.astype(np.float32)

        # ----- RK4 integration for state -----
        # s' = f(s, a), with a constant over [t, t+dt]
        k1 = self._dynamics(s, action)
        k2 = self._dynamics(s + 0.5 * self.dt * k1, action)
        k3 = self._dynamics(s + 0.5 * self.dt * k2, action)
        k4 = self._dynamics(s + self.dt * k3, action)

        s_next = s + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Optional: clip speed AFTER integration
        x, y, vx, vy = s_next
        speed = np.sqrt(vx * vx + vy * vy)
        if speed > self.max_speed:
            vx *= self.max_speed / (speed + 1e-8)
            vy *= self.max_speed / (speed + 1e-8)
            s_next = np.array([x, y, vx, vy], dtype=np.float32)

        self.state = s_next
        self.step_count += 1

        # Check boundary
        x, y, vx, vy = self.state
        r = np.sqrt(x * x + y * y)
        out_of_bounds = r > self.radius + 1e-6

        done = False
        if self.terminate_on_out_of_bounds and out_of_bounds:
            done = True
        if self.step_count >= self.max_steps:
            done = True

        # Default: sandbox dynamics, no task reward
        reward = 0.0

        info = {
            "radius": float(r),
            "out_of_bounds": bool(out_of_bounds),
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.clip(self.state, self.observation_space.low, self.observation_space.high)

    def _render_frame(self):
        x, y, _, _ = self.state

        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots()
            self._ax.set_aspect("equal", adjustable="box")
            self._ax.set_xlim(-self.radius, self.radius)
            self._ax.set_ylim(-self.radius, self.radius)

            circle = plt.Circle((0, 0), self.radius, fill=False)
            self._ax.add_patch(circle)
            (self._agent_marker,) = self._ax.plot([], [], "ro")

            plt.ion()
            plt.show()

        self._agent_marker.set_data([x], [y])
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        pass


# ------------- Registration with Gym -------------

# The id string follows the pattern of the other 2D navigation envs
# in the gym-extensions paper (Navigation-2d-Map{..}-Goal{..}-v0),
# but with a simpler name for a circular sandbox.
try:
    register(
        id="Circle-World-Double-Integrator-v0",
        entry_point="gym_extensions.continuous.circle_world_double_integrator:CircleWorldDoubleIntegrator",
        max_episode_steps=200,
    )
except gym.error.Error:
    # If already registered, ignore
    pass
