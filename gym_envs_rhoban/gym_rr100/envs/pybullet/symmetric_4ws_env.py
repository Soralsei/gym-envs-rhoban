from typing import Iterable
import numpy as np
import pybullet as p
import math

from gym_envs_rhoban.base import GoalSpaceSize
from gym_envs_rhoban.gym_rr100.envs.pybullet.rr100_env import (
    RR100ReachEnv,
    GoalPositionMixin,
    GoalPoseMixin,
)

from gymnasium import spaces

class Symmetric4WSReachEnv(RR100ReachEnv):
    MAX_PHI_C: float = 0.347

    def _load_robot(self):
        super()._load_robot()
        self.robot_limits = np.array(
            [
                np.pi,
                Symmetric4WSReachEnv.MAX_PHI_C,
            ]
        )

    def _set_action(self, action):
        assert action.shape == (self.n_actions,), "Action shape error"
        action = np.clip(action, self.action_space.low, self.action_space.high)
        clipped_action = np.clip(
            action * self.robot_limits,
            -self.robot_limits,
            self.robot_limits,
        )
        smoothed_action = self.limit_action(
            clipped_action,
            self.previous_action,
            self.robot_acceleration_limits,
            self.action_dt,
        )
        side = 1 if smoothed_action[1] > 0 else -1
        phi_i = math.atan2(
            (2 * self.wheel_base * math.sin(smoothed_action[1])),
            (
                2 * self.wheel_base * math.cos(smoothed_action[1])
                - side * self.steering_track * math.sin(smoothed_action[1])
            ),
        )
        phi_o = math.atan2(
            (2 * self.wheel_base * math.sin(smoothed_action[1])),
            (
                2 * self.wheel_base * math.cos(smoothed_action[1])
                + side * self.steering_track * math.sin(smoothed_action[1])
            ),
        )
        R_steering_i = self.wheel_base * math.tan(np.pi / 2 - abs(phi_i))
        R_steering_o = R_steering_i + self.steering_track
        R_steering_c = R_steering_i + (self.steering_track / 2)

        R_w_i = np.sqrt(R_steering_i**2 + self.wheel_base**2)
        R_w_o = np.sqrt(R_steering_o**2 + self.wheel_base**2)
        R_w_c = np.sqrt(R_steering_c**2 + self.wheel_base**2)

        w_i = smoothed_action[0] * R_w_i / R_w_c
        w_o = smoothed_action[0] * R_w_o / R_w_c

        if smoothed_action[1] > 0:
            velocities = [w_i, w_o, w_i, w_o]
            positions = [phi_i, phi_o, -phi_i, -phi_o]
        else:
            velocities = [w_o, w_i, w_o, w_i]
            positions = [phi_o, phi_i, -phi_o, -phi_i]

        p.setJointMotorControlArray(
            self.robot_id,
            self.wheel_joint_ids,
            p.VELOCITY_CONTROL,
            targetVelocities=velocities,
            forces=[20, 20, 20, 20],
        )

        for joint, position, force, velocity_limit in zip(
            self.steering_joint_ids,
            positions,
            self.steering_joint_forces,
            self.steering_velocity_limits,
        ):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=position,
                force=force,
                maxVelocity=velocity_limit,
            )

        self.previous_action = smoothed_action


class Symmetric4WSGoalReachEnv(GoalPositionMixin, Symmetric4WSReachEnv):
    pass


class Symmetric4WSGoalPoseEnv(GoalPoseMixin, Symmetric4WSReachEnv):
    def __init__(
        self,
        n_actions: int = 2,
        render_mode="human",
        goal_space_size: GoalSpaceSize = GoalSpaceSize.SMALL,
        distance_threshold: float = 0.1,
        angle_threshold: float = 0.2,
        agent_action_frequency: int = 40,
        wheel_acceleration_limit: float = 2 * np.pi,
        steering_acceleration_limit: float = np.pi / 6,
        error_bias: Iterable[float] = np.array([1, 1, 1]),
        should_load_walls: bool = True,
        should_reset_robot_position: bool = True,
        should_retransform_to_local: bool = False,
        physics_timestep: float = 1 / 500,
        n_substeps: int = 2,
    ):
        self.pose_of_interest = np.zeros((3,))
        self.angle_threshold = angle_threshold
        super().__init__(
            n_actions,
            render_mode,
            goal_space_size,
            distance_threshold,
            agent_action_frequency,
            wheel_acceleration_limit,
            steering_acceleration_limit,
            error_bias,
            should_load_walls,
            should_reset_robot_position,
            should_retransform_to_local,
            physics_timestep,
            n_substeps,
        )

    def _get_obs(self):
        super()._get_obs()
        return

    def _set_goal_space(self):
        self.goal_spaces = []

        # SMALL
        x_down = -2
        x_up = 2

        y_down = -2
        y_up = 2

        # Smaller goal space for training, for better generalization evaluation
        self.goal_spaces.append(
            spaces.Box(
                low=np.array([x_down, y_down, -np.pi]),
                high=np.array([x_up, y_up, np.pi]),
                dtype=np.float64,
            )
        )

        # MEDIUM
        x_down = -3.0
        x_up = 3.0

        y_down = -3.0
        y_up = 3.0

        self.goal_spaces.append(
            spaces.Box(
                low=np.array([x_down, y_down, -np.pi]),
                high=np.array([x_up, y_up, np.pi]),
                dtype=np.float64,
            )
        )

        # LARGE
        x_down = -4.0
        x_up = 4.0

        y_down = -4.0
        y_up = 4.0

        self.goal_spaces.append(
            spaces.Box(
                low=np.array([x_down, y_down, -np.pi]),
                high=np.array([x_up, y_up, np.pi]),
                dtype=np.float64,
            )
        )

        if self.should_load_walls:
            self._load_walls(x_down, x_up, y_down, y_up)

        self.position_space = spaces.Box(
            low=np.array([x_down, y_down, 0], dtype=np.float64),
            high=np.array([x_up, y_up, 0], dtype=np.float64),
            dtype=np.float64,
        )
