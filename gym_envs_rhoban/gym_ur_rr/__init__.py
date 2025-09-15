# import logging
from gymnasium import register

register(
    id="WholeBodyControl-v0",
    entry_point="gym_envs_rhoban.gym_ur_rr.envs:WholeBodyControlEnv",
)
register(
    id="WholeBodyControlGoal-v0",
    entry_point="gym_envs_rhoban.gym_ur_rr.envs:WholeBodyControlGoalEnv",
)
