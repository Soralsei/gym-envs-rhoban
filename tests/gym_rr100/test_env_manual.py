import time

import gymnasium as gym
import numpy as np

import gym_envs_rhoban


import pybullet as p

if __name__ == "__main__":
    env = gym.make("RR100Reach-v0", max_episode_steps=800, )
    env.reset(seed=1, options=None)

    goals = [
        np.zeros(2),
        np.array([3, 0]),
        np.array([-3, 0]),
        np.array([3, 3]),
        np.array([-3, -3]),
    ]
    
    actions = [9.523] * 4
    env.reset()
    
    p.setJointMotorControlArray(
        env.unwrapped.robot_id,
        env.unwrapped.wheel_joint_ids,
        p.VELOCITY_CONTROL,
        targetVelocities=actions,
        forces=[20, 20, 20, 20],
    )
    try:
        while True:
            for _ in range(240):
                p.stepSimulation()
                time.sleep(1/240)
            print(p.getLinkState(env.unwrapped.robot_id, 0))
            input("Press enter to continues...")
            env.reset()
    except KeyboardInterrupt:
        pass

            
    
