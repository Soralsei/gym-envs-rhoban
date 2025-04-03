import time

import gymnasium as gym
import numpy as np

import gym_envs_rhoban
from gym_envs_rhoban.gym_rr100.wrappers import ExponentialAngleRewardWrapper


def print_info(action, obs, reward, terminated, truncated, info):
    print(f"Action : {action}")
    print(f"Observation : {obs}")
    print(f"Reward : {reward}")
    print(f"Terminated : {terminated}")
    print(f"Truncated : {truncated}")
    print(f"Info : {info}")

if __name__ == "__main__":
    env = gym.make("RR100Reach-v0", max_episode_steps=800)
    env = ExponentialAngleRewardWrapper(env, theta_coeff=3.0)
    env.reset(seed=1, options=None)
    
    goals = [np.array([1, 0]), np.array([0, 1]), np.ones(2), -np.ones(2), np.array([1, -1]), np.array([-1, 1])]
    
    action = np.zeros(env.unwrapped.action_space.shape)
    for goal in goals:
        print(f"Goal : {goal}")
        env.unwrapped.goal = goal
        all = env.step(action)
        print_info(action, *all)
        
        input("Press Enter to continue...")
