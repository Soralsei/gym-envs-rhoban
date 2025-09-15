import numpy as np
import pybullet as p

from typing import Iterable, TypeVar
from gymnasium import RewardWrapper

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class FanRewardWrapper(RewardWrapper):
    
    def __init__(self, env, delta_weight: float = 1.0, error_bias: Iterable[float] = [1.0, 1.0]):
        super().__init__(env)
        self.delta_weight = delta_weight
        self.error_bias = np.array(error_bias)
    
    def reward(self, reward: float) -> float:
        goal_robot_frame = self.env.unwrapped.goal_robot_frame

        xy = np.abs(goal_robot_frame[:2])
        delta = min(xy[0] - xy[1], 0)

        goal_robot_frame[1] -= np.sign(goal_robot_frame[1]) * delta * self.delta_weight
        reward = -np.linalg.norm(goal_robot_frame * self.error_bias)

        return reward