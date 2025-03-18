import numpy as np

from typing import SupportsFloat
from gymnasium import RewardWrapper


class SquareAngleRewardWrapper(RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        print(f"Wrapping env with an SquareAngleRewardWrapper")

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        goal_robot_frame = self.env.unwrapped.goal_robot_frame
        theta = np.arctan2(goal_robot_frame[1], goal_robot_frame[0])

        return np.exp(reward) + np.exp(-(theta ** 2))
