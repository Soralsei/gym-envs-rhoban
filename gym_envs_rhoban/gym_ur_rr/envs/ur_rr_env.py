from enum import IntEnum
import os
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.core.multiarray import array as array
import pybullet as p
import pybullet_data as pd

from gym_envs_rhoban import get_urdf_path

from gym_envs_rhoban.gym_rr100.envs.pybullet import (
    AckermannGoalReachEnv,
    AckermannReachEnv,
)
from gym_envs_rhoban.base import GoalSpaceSize
from gym_envs_rhoban.gym_rr100.envs.pybullet.rr100_env import GoalPositionMixin


class WholeBodyControlEnv(AckermannReachEnv):

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    UR_CONTROLLABLE_JOINTS: list[str] = [
        "ur_shoulder_pan_joint",
        "ur_shoulder_lift_joint",
        "ur_elbow_joint",
        "ur_wrist_1_joint",
        "ur_wrist_2_joint",
        # "ur_wrist_3_joint",
    ]

    def __init__(
        self,
        render_mode="human",
        goal_space_size: GoalSpaceSize = GoalSpaceSize.SMALL,
        distance_threshold: float = 0.1,
        agent_action_frequency: int = 40,
        arm_eef_joint_name="ur_ee_link_joint",
        arm_joint_acceleration_limit: float = 2 * np.pi,
        linear_velocity_limit: float = 2,
        linear_acceleration_limit: float = 2.5,
        angular_velocity_limit: float = 1,
        angular_acceleration_limit: float = 3,
        should_load_walls: bool = True,
        should_reset_robot_position: bool = True,
        should_retransform_to_local: bool = False,
        physics_timestep: float = 1 / 240,
        n_substeps: int = 20,
    ):
        self.arm_joint_acceleration_limit = arm_joint_acceleration_limit
        self.arm_eef_joint_name = arm_eef_joint_name

        super().__init__(
            render_mode=render_mode,
            n_actions=2 + len(WholeBodyControlEnv.UR_CONTROLLABLE_JOINTS),
            goal_space_size=goal_space_size,
            distance_threshold=distance_threshold,
            agent_action_frequency=agent_action_frequency,
            linear_velocity_limit=linear_velocity_limit,
            linear_acceleration_limit=linear_acceleration_limit,
            angular_velocity_limit=angular_velocity_limit,
            angular_acceleration_limit=angular_acceleration_limit,
            should_load_walls=should_load_walls,
            should_reset_robot_position=should_reset_robot_position,
            should_retransform_to_local=should_retransform_to_local,
            physics_timestep=physics_timestep,
            n_substeps=n_substeps,
        )

    def _set_action(self, action):
        assert action.shape == (
            self.n_actions,
        ), f"Action shape {action.shape} is not valid, expected ({self.n_actions},)"
        super()._set_action(action)
        smoothed_action = self.previous_action

        for joint, position, force, velocity_limit in zip(
            self.ur_joint_ids,
            smoothed_action[2:],
            self.ur_joint_forces,
            self.ur_velocity_limits,
        ):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=position,
                force=force,
                maxVelocity=velocity_limit / 4,
            )

    def _get_obs(self):
        obs = super()._get_obs()

        ur_joint_states = p.getJointStates(
            self.robot_id,
            [
                self.robot_joint_info[joint][0]
                for joint in WholeBodyControlEnv.UR_CONTROLLABLE_JOINTS
            ],
        )
        ur_joint_positions = np.array([state[0] for state in ur_joint_states])
        ur_joint_velocities = np.array([state[1] for state in ur_joint_states])
        ee_state = p.getLinkState(self.robot_id, self.arm_eef_id, computeLinkVelocity=1)
        self.gripper_position = np.array(ee_state[0])
        self.pos_of_interest = self.gripper_position.copy()
        self.gripper_orientation = ee_state[1]
        self.gripper_velocity = np.array(ee_state[6])

        obs = np.concatenate(
            (
                obs,
                ur_joint_positions,
                ur_joint_velocities,
                self.gripper_position,
                self.gripper_velocity,
            )
        )

        return obs

    def _reset_robot(self, reset_position):
        super()._reset_robot(reset_position)
        self._set_initial_joints_positions

    def _sample_goal(self, goal_space, allow_out_of_bounds=False):
        goal = np.array(goal_space.sample())
        p.resetBasePositionAndOrientation(self.sphere, goal, self.start_orientation)
        return goal.copy()

    def _is_success(self, distance_error):
        return distance_error < self.distance_threshold

    def _load_robot(self):
        path = os.path.join(get_urdf_path(), "UR5_RR100/rr100_ur.urdf")
        self.robot_id = p.loadURDF(
            path,
            basePosition=[0, 0, 0],
            baseOrientation=self.start_orientation,
            # useFixedBase=self.use_ur_only,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_IMPLICIT_CYLINDER,
        )

        num_joint = p.getNumJoints(self.robot_id)  # 26 joints
        joint_infos = [p.getJointInfo(self.robot_id, id) for id in range(num_joint)]
        self.robot_joint_info = {
            joint_info[1].decode(): joint_info[0:1] + joint_info[2:]
            for joint_info in joint_infos
        }

        for joint_name, joint_info in self.robot_joint_info.items():
            print(
                f"Joint: {joint_name}/{joint_info[0]}, joint type: {WholeBodyControlEnv.JOINT_TYPE_STR[joint_info[1]]}, limits: {(joint_info[7], joint_info[8])}"
            )

        self.arm_eef_id = self.robot_joint_info[self.arm_eef_joint_name][0]
        self.ur_joint_ids = [
            self.robot_joint_info[joint][0]
            for joint in WholeBodyControlEnv.UR_CONTROLLABLE_JOINTS
        ]
        self.ur_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in WholeBodyControlEnv.UR_CONTROLLABLE_JOINTS
        ]
        self.wheel_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in WholeBodyControlEnv.RR_VELOCITY_JOINTS
        ]
        self.steering_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in WholeBodyControlEnv.RR_POSITION_JOINTS
        ]
        self.wheel_joint_ids = [
            self.robot_joint_info[joint][0]
            for joint in WholeBodyControlEnv.RR_VELOCITY_JOINTS
        ]
        print(f"Wheel joint ids: {self.wheel_joint_ids}")
        self.steering_joint_ids = [
            self.robot_joint_info[joint][0]
            for joint in WholeBodyControlEnv.RR_POSITION_JOINTS
        ]

        self.ur_velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in WholeBodyControlEnv.UR_CONTROLLABLE_JOINTS
            ]
        )
        self.ur_position_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in WholeBodyControlEnv.UR_CONTROLLABLE_JOINTS
            ]
        )

        self.wheel_velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in WholeBodyControlEnv.RR_VELOCITY_JOINTS
            ]
        )
        self.steering_position_limits = np.array(
            [
                (
                    self.robot_joint_info[joint][8 - 1],
                    self.robot_joint_info[joint][9 - 1],
                )
                for joint in WholeBodyControlEnv.RR_POSITION_JOINTS
            ]
        )
        self.steering_velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in WholeBodyControlEnv.RR_POSITION_JOINTS
            ]
        )

        self.robot_limits = np.array(
            [
                self.linear_velocity_limit,
                self.angular_velocity_limit,
                *self.ur_velocity_limits,
            ]
        )
        self.robot_acceleration_limits = np.array(
            [
                self.linear_acceleration_limit,
                self.angular_acceleration_limit,
                *np.full_like(
                    self.ur_velocity_limits, self.arm_joint_acceleration_limit
                ),
            ]
        )

        self.ur_base_link_id = self.robot_joint_info["ur_base_joint"][0]

        for joint in WholeBodyControlEnv.RR_POSITION_JOINTS:
            p.setCollisionFilterPair(
                self.robot_id,
                self.robot_id,
                self.robot_joint_info["base_joint"][0],
                self.robot_joint_info[joint][0],
                0,
            )
        for joint in WholeBodyControlEnv.RR_VELOCITY_JOINTS:
            p.setCollisionFilterPair(
                self.robot_id,
                self.robot_id,
                self.robot_joint_info["base_joint"][0],
                self.robot_joint_info[joint][0],
                0,
            )

    def _set_initial_joints_positions(self, init_gripper=True):
        super()._set_initial_joints_positions()
        initial_positions = {
            "ur_shoulder_pan_joint": np.pi,
            "ur_shoulder_lift_joint": -np.pi / 2,
            "ur_elbow_joint": np.pi / 2,
            "ur_wrist_1_joint": -np.pi / 2,
            "ur_wrist_2_joint": -np.pi / 2,
            "ur_wrist_3_joint": 0.0,
        }

        indices = (
            self.robot_joint_info[joint][0] for joint in initial_positions.keys()
        )

        for id, position in zip(indices, initial_positions.values()):
            p.resetJointState(self.robot_id, id, position)

    def _set_goal_space(self):
        self.goal_spaces = []

        # SMALL
        x_down = -2.0
        x_up = 2.0

        y_down = -2.0
        y_up = 2.0

        z_down = 0.8
        z_up = 1.0

        # Smaller goal space for training, for better generalization evaluation
        self.goal_spaces.append(
            spaces.Box(
                low=np.array([x_down, y_down, z_down]),
                high=np.array([x_up, y_up, z_up]),
            )
        )

        # MEDIUM
        x_down = -3.0
        x_up = 3.0

        y_down = -3.0
        y_up = 3.0

        z_down = 0.75
        z_up = 1.15

        self.goal_spaces.append(
            spaces.Box(
                low=np.array([x_down, y_down, z_down]),
                high=np.array([x_up, y_up, z_up]),
            )
        )

        # LARGE
        x_down = -4.0
        x_up = 4.0

        y_down = -4.0
        y_up = 4.0

        z_down = 0.64
        z_up = 1.34

        self.goal_spaces.append(
            spaces.Box(
                low=np.array([x_down, y_down, z_down]),
                high=np.array([x_up, y_up, z_up]),
            )
        )

        self.position_space = spaces.Box(
            low=np.array([x_down, y_down, z_down], dtype=np.float64),
            high=np.array([x_up, y_up, z_up], dtype=np.float64),
        )

        if self.should_load_walls:
            self._load_walls(x_down, x_up, y_down, y_up)

    def debug_gui(self, low, high, color=[0, 0, 1]):
        low_array = low
        high_array = high

        p1 = [low_array[0], low_array[1], low_array[2]]  # xmin, ymin, zmin
        p2 = [high_array[0], low_array[1], low_array[2]]  # xmax, ymin, zmin
        p3 = [high_array[0], high_array[1], low_array[2]]  # xmax, ymax, zmin
        p4 = [low_array[0], high_array[1], low_array[2]]  # xmin, ymax, zmin

        p.addUserDebugLine(p1, p2, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p2, p3, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p3, p4, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p4, p1, lineColorRGB=color, lineWidth=2.0, lifeTime=0)

        p5 = [low_array[0], low_array[1], high_array[2]]  # xmin, ymin, zmax
        p6 = [high_array[0], low_array[1], high_array[2]]  # xmax, ymin, zmax
        p7 = [high_array[0], high_array[1], high_array[2]]  # xmax, ymax, zmax
        p8 = [low_array[0], high_array[1], high_array[2]]  # xmin, ymax, zmax

        p.addUserDebugLine(p5, p6, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p6, p7, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p7, p8, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p8, p5, lineColorRGB=color, lineWidth=2.0, lifeTime=0)

        p.addUserDebugLine(p1, p5, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p2, p6, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p3, p7, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p4, p8, lineColorRGB=color, lineWidth=2.0, lifeTime=0)


class WholeBodyControlGoalEnv(GoalPositionMixin, WholeBodyControlEnv):
    pass
