import time

import gymnasium as gym
import numpy as np

import gym_envs_rhoban
# from gym_envs_rhoban.gym_rr100.wrappers import TerminateIfOutOfBounds

import pybullet as p

from gym_envs_rhoban.gym_rr100.wrappers.pybullet.single_obstacle import SingleObstacleWrapper


if __name__ == "__main__":
    env = gym.make("Symmetric4WSReach-v1", max_episode_steps=800, should_load_walls=False)
    env = SingleObstacleWrapper(env)
    # env = TerminateIfOutOfBounds(env)
    env.reset(seed=1, options=None)
    input("Press enter to continue...")
    
    actions = np.array(
        [
            (1.0, 0, env.unwrapped.robot_action_frequency),
            (-1.0, 0, env.unwrapped.robot_action_frequency),
            (1.0, 1.0, env.unwrapped.robot_action_frequency),
            (-1.0, 1.0, env.unwrapped.robot_action_frequency),
            (1.0, -1.0, env.unwrapped.robot_action_frequency),
            (-1.0, -1.0, env.unwrapped.robot_action_frequency),
        ]
    )
    
    for s, a, freq in actions:
        obs, info = env.reset()
        print(f"Initial observation : {obs}")
        print(f"Initial info : {info}")
        action = np.array([s, a])
        env.unwrapped.manual_set_goal([1, 0])
        for _ in range(int(freq * 2)):
            obs, reward, terminated, truncated, info =  env.step(action)
            time.sleep(env.unwrapped.action_dt)
            if terminated or truncated:
                break
        print(obs)
        input("Press enter to continues...")
            
    
