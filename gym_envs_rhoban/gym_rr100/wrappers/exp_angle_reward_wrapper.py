import numpy as np

from typing import SupportsFloat
from gymnasium import RewardWrapper


class ExponentialAngleRewardWrapper(RewardWrapper):

    def __init__(self, env, theta_coeff: float=1.0):
        super().__init__(env)
        self.theta_coeff = theta_coeff = theta_coeff
        print(f"Wrapping env with an {self.class_name()}")

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        goal_robot_frame = self.env.unwrapped.goal_robot_frame
        theta = np.arctan2(goal_robot_frame[1], goal_robot_frame[0])

        return np.exp(reward) + self.theta_coeff * np.exp(-(theta ** 2))
