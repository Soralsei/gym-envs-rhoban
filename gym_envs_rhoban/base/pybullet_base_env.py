from abc import abstractmethod
from typing import Iterable
from gym_envs_rhoban.base import BaseEnv
import gymnasium as gym
from gymnasium import spaces as spaces
import pybullet as p
import pybullet_data as pd


class PyBulletBaseEnv(BaseEnv):

    JOINT_TYPE_STR: dict = {
        p.JOINT_REVOLUTE: "revolute",
        p.JOINT_PRISMATIC: "prismatic",
        p.JOINT_SPHERICAL: "spherical",
        p.JOINT_PLANAR: "planar",
        p.JOINT_FIXED: "fixed",
    }

    def __init__(
        self,
        n_actions: int,
        render_mode="human",
        agent_action_frequency=40,
        physics_step: float = 1 / 240,
        n_substeps: int = 10,
        gravity_constant: float = -9.81,
    ):
        super().__init__(n_actions, render_mode, agent_action_frequency)

        self.physics_step = physics_step
        self.n_substeps = n_substeps
        self.gravity_constant = gravity_constant

        # connect bullet
        simulation_type = p.DIRECT if render_mode != "human" else p.GUI
        p.connect(simulation_type)

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.cam_width = 800
        self.cam_height = 640

    def _init_simulation(self):
        p.resetSimulation()

        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.physics_step)
        p.setPhysicsEngineParameter(numSubSteps=self.n_substeps)

        p.setGravity(0, 0, self.gravity_constant)

    def _step_simulation(self):
        for _ in range(int((1 / self.physics_step) // self.robot_action_frequency)):
            p.stepSimulation()

    def _close(self):
        p.disconnect()

    def _render_human(self):
        return None

    def _render_rgb_array(self):
        return p.getCameraImage(
            self.cam_width,
            self.cam_height,
            self._get_camera_view_matrix(),
            self._get_camera_projection_matrix(),
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )[2]

    @abstractmethod
    def _get_camera_view_matrix(self):
        pass

    @abstractmethod
    def _get_camera_projection_matrix(self):
        pass
