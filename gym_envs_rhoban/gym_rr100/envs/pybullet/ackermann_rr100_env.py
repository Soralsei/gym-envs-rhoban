from typing import Iterable
import numpy as np
import pybullet as p

from gymnasium import spaces as spaces

from .rr100_env import RR100ReachEnv, GoalPositionMixin
from gym_envs_rhoban.base import GoalSpaceSize


class AckermannReachEnv(RR100ReachEnv):

    def __init__(
        self,
        render_mode="human",
        n_actions: int = 2,
        goal_space_size: GoalSpaceSize = GoalSpaceSize.SMALL,
        distance_threshold: float = 0.10,
        agent_action_frequency: int = 40,
        linear_velocity_limit: float = 2.0,
        linear_acceleration_limit: float = 2.5,
        angular_velocity_limit: float = 1.0,
        angular_acceleration_limit: float = 3.0,
        error_bias: Iterable[float] = np.array([1.0, 1.0, 1.0]),
        should_load_walls: bool = True,
        should_reset_robot_position: bool = True,
        should_retransform_to_local: bool = False,
        physics_timestep: float = 1 / 240.0,
        n_substeps: int = 1,
    ):
        self.linear_velocity_limit = linear_velocity_limit
        self.angular_velocity_limit = angular_velocity_limit
        self.linear_acceleration_limit = linear_acceleration_limit
        self.angular_acceleration_limit = angular_acceleration_limit

        super().__init__(
            n_actions=n_actions,
            render_mode=render_mode,
            goal_space_size=goal_space_size,
            distance_threshold=distance_threshold,
            agent_action_frequency=agent_action_frequency,
            error_bias=error_bias,
            should_load_walls=should_load_walls,
            should_reset_robot_position=should_reset_robot_position,
            should_retransform_to_local=should_retransform_to_local,
            physics_timestep=physics_timestep,
            n_substeps=n_substeps,
        )

    def _load_robot(self):
        super()._load_robot()
        self.robot_limits = np.array(
            [self.linear_velocity_limit, self.angular_velocity_limit]
        )

        self.robot_acceleration_limits = np.array(
            [self.linear_acceleration_limit, self.angular_acceleration_limit]
        )

    def _set_action(self, action):
        assert action.shape == (
            self.n_actions,
        ), f"Action shape {action.shape} is not valid, expected ({self.n_actions},)"
        action = np.clip(action, self.action_space.low, self.action_space.high)  # type: ignore

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

        vel_left_front = 0
        vel_right_front = 0
        vel_left_rear = 0
        vel_right_rear = 0

        sign = np.sign(smoothed_action[0])
        vel_left_front = (
            sign
            * np.hypot(
                smoothed_action[0] - smoothed_action[1] * self.steering_track / 2,
                (self.wheel_base * smoothed_action[1] / 2.0),
            )
            / self.wheel_radius
        )

        vel_right_front = (
            sign
            * np.hypot(
                smoothed_action[0] + smoothed_action[1] * self.steering_track / 2,
                (self.wheel_base * smoothed_action[1] / 2.0),
            )
            / self.wheel_radius
        )

        vel_left_rear = (
            sign
            * np.hypot(
                smoothed_action[0] - smoothed_action[1] * self.steering_track / 2,
                (self.wheel_base * smoothed_action[1] / 2.0),
            )
            / self.wheel_radius
        )
        vel_right_rear = (
            sign
            * np.hypot(
                smoothed_action[0] + smoothed_action[1] * self.steering_track / 2,
                (self.wheel_base * smoothed_action[1] / 2.0),
            )
            / self.wheel_radius
        )

        front_left_steering = 0
        front_right_steering = 0
        rear_left_steering = 0
        rear_right_steering = 0

        if abs(2.0 * smoothed_action[0]) > abs(
            smoothed_action[1] * self.steering_track
        ):
            front_left_steering = np.arctan(
                smoothed_action[1]
                * self.wheel_base
                / (2.0 * smoothed_action[0] - smoothed_action[1] * self.steering_track)
            )
            front_right_steering = np.arctan(
                smoothed_action[1]
                * self.wheel_base
                / (2.0 * smoothed_action[0] + smoothed_action[1] * self.steering_track)
            )
        elif abs(smoothed_action[0] > 1e-3):
            sign = np.sign(smoothed_action[1])
            front_left_steering = sign * AckermannReachEnv.M_PI_2
            front_right_steering = sign * AckermannReachEnv.M_PI_2

        rear_left_steering = -front_left_steering
        rear_right_steering = -front_right_steering

        velocities = [vel_left_front, vel_right_front, vel_left_rear, vel_right_rear]
        p.setJointMotorControlArray(
            self.robot_id,
            self.wheel_joint_ids,
            p.VELOCITY_CONTROL,
            targetVelocities=velocities,
            forces=[20, 20, 20, 20],
        )

        positions = [
            front_left_steering,
            front_right_steering,
            rear_left_steering,
            rear_right_steering,
        ]
        # Didn't respect velocity limits ??
        p.setJointMotorControlArray(
            self.robot_id,
            self.steering_joint_ids,
            p.POSITION_CONTROL,
            targetPositions=positions,
            forces=self.steering_joint_forces,
            # maxVelocities=self.steering_velocity_limits
        )
        # for joint, position, force, velocity_limit in zip(
        #     self.steering_joint_ids,
        #     positions,
        #     self.steering_joint_forces,
        #     self.steering_velocity_limits,
        # ):
        #     print(f"Setting joint {joint} to position {position} with force {force} and velocity limit {velocity_limit}")
        #     p.setJointMotorControl2(
        #         self.robot_id,
        #         joint,
        #         p.POSITION_CONTROL,
        #         targetPosition=position,
        #         force=force,
        #         maxVelocity=velocity_limit,
        #     )

        self.previous_action = smoothed_action


class AckermannGoalReachEnv(GoalPositionMixin, AckermannReachEnv):
    pass
