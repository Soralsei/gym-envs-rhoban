import time
import gymnasium as gym
import numpy as np
from gym_envs_rhoban.gym_rr100.envs import AckermannReachEnv
from gym_envs_rhoban.gym_rr100.wrappers import FastRLapWrapper


def print_info(action, obs, reward, terminated, truncated, info):
    print(f"Action : {action}")
    print(f"Observation : {obs}")
    print(f"Reward : {reward}")
    print(f"Terminated : {terminated}")
    print(f"Truncated : {truncated}")
    print(f"Info : {info}")

if __name__ == "__main__":
    env = AckermannReachEnv()
    env = FastRLapWrapper(env)
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
            (-1, 0, env.unwrapped.robot_action_frequency),
            (1, 1, env.unwrapped.robot_action_frequency),
            (-1, -1, env.unwrapped.robot_action_frequency),
        ]
    )
    env.unwrapped.goal = np.zeros(2)
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    print_info(action, obs, reward, terminated, truncated, info)
    
    for i in range(1, len(goals)):
        env.reset()
        env.unwrapped.goal = goals[i]
        
        for w, a, freq in actions:
            print(f"Goal : {goals[i]}")
            action = np.array([w,a])
            for _ in range(freq):
                obs, reward, terminated, truncated, info =  env.step(action)
                time.sleep(env.unwrapped.action_dt)
            print_info(action, obs, reward, terminated, truncated, info)
            
    env.step()
