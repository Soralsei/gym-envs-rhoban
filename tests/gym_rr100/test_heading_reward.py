import time

import gymnasium as gym
import matplotlib
matplotlib.use("qtcairo")
import matplotlib.pyplot as plt
import numpy as np

import gym_envs_rhoban
from gym_envs_rhoban.gym_rr100.wrappers import ExponentialAngleRewardWrapper


def print_info(action, obs, reward, terminated, truncated, info):
    # print(f"Action : {action}")
    print(f"Observation : {obs}")
    print(f"Reward : {reward}")
    # print(f"Terminated : {terminated}")
    # print(f"Truncated : {truncated}")
    # print(f"Info : {info}")

if __name__ == "__main__":
    env = gym.make("RR100Reach-v0", max_episode_steps=800, render_mode = "human")
    env = ExponentialAngleRewardWrapper(env, theta_coeff=1.0)
    env.reset(seed=1, options=None)
    
    # goals = [np.array([1, 8e2]), np.array([0, 1]), np.ones(2), -np.ones(2), np.array([1, -1]), np.array([-1, 1])]
    goal = np.array([0.3, 0.02761692])
    # goal = np.array([0, 0])
    action = np.array([1, 0])
    # for goal in goals:
    env.reset()
    print(f"Goal : {goal}")
    env.unwrapped.goal = goal
    reward = env.reward(env.unwrapped.reward()[0])
    print("Reward before : ", reward)
    rewards = [reward]
    # for action in actions:
    for _ in range(40):
        all_info = env.step(action)
        print_info(action, *all_info)
        rewards.append(all_info[1])
        if all_info[2] or all_info[3]:
            break
    
    plt.plot(rewards)
    # plt.ylim(0, 1)
    plt.title("Reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.show()
    # input("Press Enter to continue...")
