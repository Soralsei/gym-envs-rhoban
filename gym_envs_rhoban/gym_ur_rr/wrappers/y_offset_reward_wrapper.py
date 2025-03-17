import warnings as w

from gymnasium import RewardWrapper

class YOffsetRewardWrapper(RewardWrapper):
    
    def __init__(self, env, safe_offset : float = 0.4):
        super().__init__(env)
        self.safe_offset = safe_offset
        w.filterwarnings("once", append=True)
    
    def reward(self, reward: float) -> float:
        try:
            y_offset = self.env.unwrapped.goal[1] - self.env.unwrapped.robot_position[1]
            y_offset = abs(y_offset)
            
            if y_offset <= self.safe_offset:
                y_offset = 0.0
        except AttributeError:
            w.warn("Missing attribute env.goal or env.robot_position, returning an unchanged reward")
            return reward

        return reward - y_offset