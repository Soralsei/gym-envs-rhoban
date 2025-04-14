# import logging
from gymnasium import register

register(
    id='AckermannGoalReach-v0',
    entry_point='gym_envs_rhoban.gym_rr100.envs:AckermannGoalEnv',
)
register(
    id='AckermannReach-v0',
    entry_point='gym_envs_rhoban.gym_rr100.envs:AckermannReachEnv',
)
register(
    id='RR100Reach-v0',
    entry_point='gym_envs_rhoban.gym_rr100.envs:RR100ReachEnv',
)
register(
    id='RR100GoalReach-v0',
    entry_point='gym_envs_rhoban.gym_rr100.envs:RR100ReachGoalEnv',
)