import numpy as np
import pybullet as p

from typing import Iterable, TypeVar
from gymnasium import RewardWrapper


class TurningRadiusReward(RewardWrapper):

    def __init__(self, env, min_turning_radius: float = 0.5, penalty: float = 5.0):
        super().__init__(env)
        self.min_turning_radius = min_turning_radius

    def reward(self, reward: float) -> float:  # type: ignore
        penalty = 0.0
        goal_in_robot_frame = np.array(self.env.unwrapped._get_position_in_robot_frame(self.env.unwrapped.goal))  # type: ignore
        dist_left = np.linalg.norm(
            goal_in_robot_frame - np.array([0, self.min_turning_radius + self.env.unwrapped.distance_threshold])
        )
        dist_right = np.linalg.norm(
            goal_in_robot_frame - np.array([0, -self.min_turning_radius - self.env.unwrapped.distance_threshold])
        )
        if dist_left < self.min_turning_radius or dist_right < self.min_turning_radius:
            penalty = 2.0
        return reward - penalty
