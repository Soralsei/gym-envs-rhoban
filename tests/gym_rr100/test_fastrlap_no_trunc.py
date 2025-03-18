import time

import gymnasium as gym
import numpy as np

import gym_envs_rhoban
from gym_envs_rhoban.gym_rr100.wrappers import NoTruncFastRLapWrapper
from gym_envs_rhoban.gym_rr100.wrappers import FastRLapWrapper


def print_info(action, obs, reward, terminated, truncated, info):
    print(f"Action : {action}")
    print(f"Observation : {obs}")
    print(f"Reward : {reward}")
    print(f"Terminated : {terminated}")
    print(f"Truncated : {truncated}")
    print(f"Info : {info}")

if __name__ == "__main__":
    env = gym.make("RR100Reach-v0", max_episode_steps=800)
    env = NoTruncFastRLapWrapper(env)
    env.reset(seed=1, options=None)
    
    action = np.zeros(env.unwrapped.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    for _ in range(800):
        obs, reward, terminated, truncated, info = env.step(action)
    
    print_info(action, obs, reward, terminated, truncated, info)
