import math
import warnings as w

import numpy as np
import pybullet as p

from typing import SupportsFloat, TypeVar, Union
from gymnasium import RewardWrapper

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class CarRewardWrapper(RewardWrapper):

    def __init__(self, env, c1: float=0.01, c2: float=1.5):
        super().__init__(env)
        print(f"Wrapping env with an {self.class_name()}")
        self.c1 = c1
        self.c2 = c2

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        try:
            goal_robot_frame = self.env.unwrapped.goal_robot_frame
            d = np.sum(np.abs(goal_robot_frame))
            theta = abs(np.arctan2(goal_robot_frame[1], goal_robot_frame[0]))
            
            return 0.01 * (-self.c1 * d**2 - self.c2 * (1 - np.cos(theta) / (d + 1)))
        except Exception:
            pass
        return reward
