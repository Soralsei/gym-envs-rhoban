import warnings as w

import numpy as np
from gymnasium import Wrapper


class FastRLapWrapper(Wrapper):

    def __init__(self, env, n_positions: int = 100):
        super().__init__(env)
        self._last_positions = np.empty((n_positions, 2), dtype=np.float32)
        self._last_positions.fill(np.inf)
        print(f"Wrapping environment with {self.class_name()}")
        w.filterwarnings("once", append=True)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        try:
            reward, timeout = self.reward()
            self._last_positions = np.roll(self._last_positions, -1, axis=0)
            self._last_positions[-1] = self.env.unwrapped.robot_position
        except AttributeError as e:
            w.warn(f"Missing attribute : {e}")
        
        return observation, reward, terminated, truncated | timeout, info

    def reward(self):
        # goal_to_robot = self.env.unwrapped.robot_position - self.env.unwrapped.goal
        # distance_to_goal = np.linalg.norm(goal_to_robot)
        
        vector_to_goal = self.env.unwrapped.goal - self.env.unwrapped.robot_position
        distance_to_goal = np.linalg.norm(vector_to_goal)
        
        normalized_vector = vector_to_goal / (distance_to_goal + 1e-6)
        distance_threshold = self.env.unwrapped.distance_threshold

        # velocity_to_goal = np.sum(self.env.unwrapped.robot_velocity * normalized_vector)
        velocity_to_goal = np.dot(self.env.unwrapped.robot_velocity, normalized_vector)

        should_timeout = self.should_timeout
        
        reward = (
            velocity_to_goal
            + 100 * (distance_to_goal < distance_threshold)
            - 100 * should_timeout
        )

        return reward, should_timeout

    def reset(self, *, seed=None, options=None):
        self._last_positions.fill(np.inf)
        return super().reset(seed=seed, options=options)

    @property
    def should_timeout(self):
        return np.linalg.norm(
                self.env.unwrapped.robot_position - self._last_positions
            ).max() < 1.0
