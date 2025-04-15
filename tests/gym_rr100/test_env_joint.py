import time

import gymnasium as gym
import numpy as np

import gym_envs_rhoban
from gym_envs_rhoban.gym_rr100.wrappers import TerminateIfOutOfBounds

import pybullet as p


if __name__ == "__main__":
    env = gym.make("RR100Reach-v0", max_episode_steps=800, should_load_walls=False, is_rl_ackermann=True)
    env = TerminateIfOutOfBounds(env)
    env.reset(seed=1, options=None)
    
    actions = np.array(
        [
            # (1.0, 0, env.unwrapped.robot_action_frequency),
            # (-1.0, 0, env.unwrapped.robot_action_frequency),
            (1.0, 1.0, env.unwrapped.robot_action_frequency),
            (-1.0, 1.0, env.unwrapped.robot_action_frequency),
            (1.0, -1.0, env.unwrapped.robot_action_frequency),
            (-1.0, -1.0, env.unwrapped.robot_action_frequency),
        ]
    )
    
    for s, a, freq in actions:
        env.reset()
        action = np.array([s, a])
        for _ in range(int(freq * 2)):
            obs, reward, terminated, truncated, info =  env.step(action)
            time.sleep(env.unwrapped.action_dt)
            if terminated or truncated:
                break
        print(p.getLinkState(env.unwrapped.robot_id, 0))
        input("Press enter to continues...")
            
    
