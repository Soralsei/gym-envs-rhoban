# import logging
from gymnasium import register

register(
    id='UrRR100Reach-v0',
    entry_point='gym_envs.gym_ur_rr.envs:UrRrReachEnv',
)