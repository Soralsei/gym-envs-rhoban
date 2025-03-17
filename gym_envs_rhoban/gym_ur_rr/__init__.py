# import logging
from gymnasium import register

register(
    id='UrRR100Reach-v0',
    entry_point='gym_envs_rhoban.gym_ur_rr.envs:UrRrReachEnv',
)