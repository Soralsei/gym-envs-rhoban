import warnings as w

import numpy as np
from gymnasium import Wrapper


class OldFastRLapRewardWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self._last_positions = np.empty((500, 2), dtype=np.float32)
        self._last_positions.fill(np.inf)

        print(f"Wrapping environment with {self.class_name()}")

        w.filterwarnings("once", append=True)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        try:
            reward = self.reward()
            self._last_positions = np.roll(self._last_positions, -1)
            self._last_positions[-1] = self.env.unwrapped.robot_position
        except AttributeError as e:
            w.warn(f"Missing attribute : {e}")

        return observation, reward, terminated, truncated, info

    def reward(self):
        vector_to_goal = self.env.unwrapped.goal - self.env.unwrapped.robot_position
        distance_to_goal = np.linalg.norm(vector_to_goal)
        normalized_vector = vector_to_goal / (distance_to_goal + 1e-6)
        velocity_to_goal = np.dot(self.env.unwrapped.robot_velocity, normalized_vector)

        distance_threshold = self.env.unwrapped.distance_threshold
        should_timeout = self.should_timeout

        # print(should_timeout)

        # print(distance_to_goal < distance_threshold)

        reward = (
            velocity_to_goal
            + 100 * (distance_to_goal < distance_threshold)
            - 100 * should_timeout
        )

        return reward

    def reset(self, *, seed=None, options=None):
        self._last_positions.fill(np.inf)
        return super().reset(seed=seed, options=options)

    @property
    def should_timeout(self):
        return (
            np.linalg.norm(
                self.env.unwrapped.robot_position - self._last_positions
            ).max()
            < 1.0
        )
