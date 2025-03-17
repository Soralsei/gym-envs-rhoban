import os
from gym_envs_rhoban import gym_panda, gym_ur_rr, gym_rr100

def get_urdf_path() -> str:
    return os.path.join(os.path.dirname(__file__), "urdf")