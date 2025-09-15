from enum import IntEnum
from .base_env import BaseEnv
from .pybullet_base_env import PyBulletBaseEnv
from .goal_env_mixin import GoalEnvMixin


class GoalSpaceSize(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
