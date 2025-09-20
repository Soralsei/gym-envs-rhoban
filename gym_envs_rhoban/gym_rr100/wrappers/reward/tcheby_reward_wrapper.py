import numpy as np

from typing import Iterable
from gymnasium import RewardWrapper

class TchebychevRewardWrapper(RewardWrapper):
    
    def __init__(self, env, error_bias: Iterable[float] = [1.0, 1.0]):
        super().__init__(env)
        print(f"Wrapping env with a {self.class_name()}")
        self.error_bias = np.array(error_bias)
    
    def reward(self, _: float) -> float:
        goal_robot_frame = np.array(self.env.unwrapped._get_position_in_robot_frame(self.env.unwrapped.goal)) # type: ignore
        goal_robot_frame = goal_robot_frame * self.error_bias
        return -np.max(np.abs(goal_robot_frame))