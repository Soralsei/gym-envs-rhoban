import math
import warnings as w

import numpy as np
import pybullet as p

from typing import SupportsFloat, TypeVar, Union
from gymnasium import RewardWrapper

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class AngleRewardWrapper(RewardWrapper):

    def __init__(self, env, nu: SupportsFloat = 1.0, beta: Union[str, float] = "auto"):
        super().__init__(env)
        print(f"Wrapping env with an AngleRewardWrapper")
        self.nu = nu
        self.beta = beta
        if isinstance(self.beta, str) and self.beta != "auto":
            w.warn(
                f"Beta parameter passed to AngleRewardWrapper '{beta}' unknown. You probably meant to pass 'auto' or a float. Using 'auto'."
            )
            self.beta = "auto"
        self.auto_beta = isinstance(self.beta, str) and self.beta == "auto"

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        try:
            goal_robot_frame = self.env.unwrapped.goal_robot_frame
            theta = abs(np.arctan2(goal_robot_frame[1], goal_robot_frame[0]))
            
            if theta >= np.pi / 2:
                theta = np.pi - theta

            if abs(goal_robot_frame[1]) <= self.env.unwrapped.distance_threshold :
                beta = 0
            else:
                if self.auto_beta:
                    dist = np.linalg.norm(goal_robot_frame)
                    beta = self.nu * math.exp(-dist + self.env.unwrapped.distance_threshold)
        except Exception:
            return reward

        return reward - theta * beta
