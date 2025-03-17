# import logging
from gymnasium import register

# logger = logging.getLogger(__name__)

register(
    id='Simple_PandaReach-v0',
    entry_point='gym_envs_rhoban.gym_panda.envs:SimplePandaReachEnv',
)

register(
    id='Joint_PandaReach-v0',
    entry_point='gym_envs_rhoban.gym_panda.envs:JointPandaReachEnv',
)

