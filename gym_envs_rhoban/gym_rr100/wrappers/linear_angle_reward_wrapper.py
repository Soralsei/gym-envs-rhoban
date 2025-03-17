import math
import warnings as w

import numpy as np
import pybullet as p

from typing import SupportsFloat, TypeVar, Union
from gymnasium import RewardWrapper

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class LinearAngleRewardWrapper(RewardWrapper):

    def __init__(self, env, max_dist:SupportsFloat = 5.0, beta: Union[str, float] = "auto"):
        super().__init__(env)
        print(f"Wrapping env with an AngleRewardWrapper")
        self.beta = beta
        self.max_dist = max_dist
        if isinstance(self.beta, str) and self.beta != "auto":
            w.warn(
                f"Beta parameter passed to AngleRewardWrapper '{beta}' unknown. You probably meant to pass 'auto' or a float. Using 'auto'."
            )
            self.beta = "auto"

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        goal_robot_frame = self.env.unwrapped.goal_robot_frame
        theta = abs(np.arctan2(goal_robot_frame[1], goal_robot_frame[0]))
        
        if theta >= np.pi / 2:
            theta = np.pi - theta

        d = self.env.unwrapped.distance_threshold
        dist = np.linalg.norm(goal_robot_frame)
        if isinstance(self.beta, str) and self.beta == "auto":
            if dist <= d:
                beta = 1
            elif dist >= self.max_dist:
                beta = 0
            else:
                slope = -1 / (self.max_dist - d)
                intercept = self.max_dist / (self.max_dist - d)
                beta = slope * dist + intercept

        return reward - theta * beta
