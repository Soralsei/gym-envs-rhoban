import gymnasium as gym
import numpy as np
import warnings

from gymnasium import Wrapper


class TerminateIfOutOfBounds(Wrapper):
    '''
    This wrapper terminates
      an episode if the agent steps out of bounds.
    
    Requires these attributes in the wrapped environment
    - `position_space: gym.Space`
    - `robot_position: np.ndarray` with the same shape as `position_space`
    '''
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # If the robot steps out of bounds, truncate the episode``
        try:
            pos_space: gym.Space = self.env.unwrapped.position_space
            should_terminate = not pos_space.contains(self.env.unwrapped.robot_position)
            if should_terminate:
                print("Terminating : robot out of bounds")
        except AttributeError:
            warnings.warn(f"Environment does not have a 'position_space' gym.Space")
            should_terminate = terminated
            
        return obs, reward, terminated | should_terminate, truncated, info