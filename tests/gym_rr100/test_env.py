import time

import gymnasium as gym
import numpy as np

import gym_envs_rhoban
from gym_envs_rhoban.gym_rr100.wrappers import TerminateIfOutOfBounds

import pybullet as p



if __name__ == "__main__":
    env = gym.make("AckermannReach-v0", max_episode_steps=800, should_load_walls=False)
    env = TerminateIfOutOfBounds(env)
    env.reset(seed=1, options=None)

    goals = [
        np.zeros(2),
        np.array([3, 0]),
        np.array([-3, 0]),
        np.array([3, 3]),
        np.array([-3, -3]),
    ]
    
    actions = np.array(
        [
            (1, 0, env.unwrapped.robot_action_frequency),
            (1, 1, env.unwrapped.robot_action_frequency),
            (1, -1, env.unwrapped.robot_action_frequency),
            (-1, 0, env.unwrapped.robot_action_frequency),
            (-1, 1, env.unwrapped.robot_action_frequency),
            (-1, -1, env.unwrapped.robot_action_frequency),
            # (0, 0, env.unwrapped.robot_action_frequency),
            # (0, 0, env.unwrapped.robot_action_frequency),
            # (0, 0, env.unwrapped.robot_action_frequency),
            # (0, 0, env.unwrapped.robot_action_frequency),
            # (0, 0, env.unwrapped.robot_action_frequency),
        ]
    )
    
    for lin, ang, freq in actions:
    # for i in range(1, len(goals)):
        env.reset()
        # env.unwrapped.goal = goals[i]
        
        # for lin, ang, freq in actions:
        # print(f"Goal : {goals[i]}")
        action = np.array([lin,ang])
        for _ in range(freq * 2):
            obs, reward, terminated, truncated, info =  env.step(action)
            time.sleep(env.unwrapped.action_dt)
            if terminated or truncated:
                break
        print(p.getLinkState(env.unwrapped.robot_id, 0))
        input("Press enter to continues...")
            
    
