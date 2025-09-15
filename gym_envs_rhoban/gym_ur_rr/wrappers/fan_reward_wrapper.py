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
        self.xy_base_offset = np.array([0.2, 0.4])
    
    def reward(self, reward: float) -> float:
        """
        try:
            y_offset = self.env.unwrapped.goal[1] - self.env.unwrapped.pos_of_interest[1]
            y_offset = abs(y_offset)
            
            if y_offset <= self.safe_offset:
                y_offset = 0.0
        except AttributeError:
            w.warn("Missing attribute env.goal or env.pos_of_interest, returning an unchanged reward")
            return reward

        return reward - y_offset
        """
        
        goal_in_ur_base_frame = self.env.unwrapped.goal_in_ur_base_frame
        # y_offset = self.env.unwrapped.goal[1] - self.env.unwrapped.pos_of_interest[1]
        # y_offset = abs(y_offset)
        
        xy = np.abs(goal_in_ur_base_frame[:2])
        xy_offset = xy + self.xy_base_offset
        delta = max(xy_offset[1] - xy_offset[0], 0)

        goal_in_ur_base_frame[1] += np.sign(goal_in_ur_base_frame[1]) * delta * self.delta_weight
        base_reward = -np.linalg.norm(goal_in_ur_base_frame * self.error_bias)

        return reward + base_reward