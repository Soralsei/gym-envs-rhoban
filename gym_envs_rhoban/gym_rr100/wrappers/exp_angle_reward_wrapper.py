import numpy as np

from typing import SupportsFloat
from gymnasium import RewardWrapper
import math


class ExponentialAngleRewardWrapper(RewardWrapper):

    def __init__(self, env, lambda_d: float=0.8, lambda_b: float=0.2, K: float = 0.5):
        super().__init__(env)
        self.lambda_b = lambda_b
        self.lambda_d = lambda_d
        self.K = K
        print(f"Wrapping env with an {self.class_name()}")

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        goal_robot_frame = self.env.unwrapped.goal_robot_frame
        theta = np.arctan2(goal_robot_frame[1], goal_robot_frame[0])

        return self.lambda_d * np.exp(self.K * reward) + self.lambda_b * np.exp(-self.K * (math.sqrt(theta ** 2)))
