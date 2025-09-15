# import logging
from gymnasium import register

# Legacy environments
register(
    id="AckermannGoalReach-v0",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet.old:AckermannGoalEnv",
)
register(
    id="AckermannReach-v0",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet.old:AckermannReachEnv",
)
register(
    id="RR100Reach-v0",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet.old:RR100ReachEnv",
)
register(
    id="RR100GoalReach-v0",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet.old:RR100ReachGoalEnv",
)

# Refactored environments
register(
    id="AckermannReach-v1",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet:AckermannReachEnv",
)
register(
    id="AckermannGoalReach-v1",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet:AckermannGoalReachEnv",
)
register(
    id="RR100Reach-v1",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet:RR100ReachEnv",
)
register(
    id="Symmetric4WSReach-v1",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet:Symmetric4WSReachEnv",
)
register(
    id="RR100GoalReach-v1",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet:RR100ReachGoalEnv",
)
register(
    id="Symmetric4WSGoalReach-v1",
    entry_point="gym_envs_rhoban.gym_rr100.envs.pybullet:Symmetric4WSGoalReachEnv",
)
