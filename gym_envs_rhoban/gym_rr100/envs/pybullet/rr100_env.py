from enum import IntEnum
import math
import os
from typing import Any, Iterable, Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data as pd

from gym_envs_rhoban import get_urdf_path
from gym_envs_rhoban.base import PyBulletBaseEnv, GoalEnvMixin, GoalSpaceSize
from gym_envs_rhoban.utils import wrapped_angle_difference


class RR100ReachEnv(PyBulletBaseEnv):

    M_PI_2 = np.pi / 2.0

    RR_VELOCITY_JOINTS: list[str] = [
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    ]
    RR_POSITION_JOINTS: list[str] = [
        "front_left_steering_joint",
        "front_right_steering_joint",
        "rear_left_steering_joint",
        "rear_right_steering_joint",
    ]

    def __init__(
        self,
        n_actions: int = 2,
        render_mode="human",
        goal_space_size: GoalSpaceSize = GoalSpaceSize.SMALL,
        distance_threshold: float = 0.10,
        agent_action_frequency: int = 40,
        wheel_acceleration_limit: float = 2 * np.pi,
        steering_acceleration_limit: float = np.pi / 6,
        error_bias: Iterable[float] = np.array([1.0, 1.0]),
        should_load_walls: bool = True,
        should_reset_robot_position: bool = True,
        should_retransform_to_local: bool = False,
        physics_timestep: float = 1 / 240.0,
        n_substeps: int = 1,
        reach_bonus: float = 0.0,
        resample_goal: bool = True,
        initial_goal: Optional[np.ndarray] = None,
    ):
        super().__init__(
            n_actions=n_actions,
            render_mode=render_mode,
            agent_action_frequency=agent_action_frequency,
            physics_step=physics_timestep,
            n_substeps=n_substeps,
        )
        self.reach_bonus = reach_bonus
        self.should_reset_robot_pos = should_reset_robot_position
        self.should_retransform_to_local = should_retransform_to_local
        self.resample_goal = resample_goal

        self.total_episodes = 0
        self.initial_distance = 0.0
        self.total_traveled = 0.0
        self.pos_of_interest = np.zeros(2)
        self.initial_robot_pose = [
            np.zeros(3),
            np.zeros(4),
        ]  # position[x,y,z], orientation[x,y,z,w]
        self.previous_action = np.zeros(self.n_actions)

        self.goal_space_size = goal_space_size
        self.should_load_walls = should_load_walls

        self.wheel_acceleration_limit = wheel_acceleration_limit
        self.steering_acceleration_limit = steering_acceleration_limit

        # robot parameters
        # TEMPORARY, should probably read from URDF or pass as argument
        self.steering_track = 0.6
        self.wheel_base = 0.5 / 2
        self.wheel_radius = 0.21

        self.distance_threshold = distance_threshold
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # error bias
        if not isinstance(error_bias, np.ndarray):
            error_bias = np.array(error_bias)
        self.error_bias = error_bias

        self._init_camera_matrices()

        self._init_simulation()

        if not resample_goal:
            self.goal = initial_goal if initial_goal is not None else self._sample_goal(
                self.goal_spaces[self.goal_space_size]
            )

        self.debug_gui(self.position_space.low, self.position_space.high, [0, 0, 1])
        p.resetDebugVisualizerCamera(
            cameraDistance=6.0,
            cameraYaw=-90,
            cameraPitch=-89,
            cameraTargetPosition=[0, 0, 0],
        )

    def _get_position_in_robot_frame(self, position: np.ndarray):
        robot_pos = [*self.pos_of_interest, 0.0]
        if position.shape == (2,):
            position = np.array([position[0], position[1], 0.0]) # if 2D, add z=0
        return self._get_pose_in_frame(
            [position, self.start_orientation],
            [robot_pos, p.getQuaternionFromEuler([0, 0, self.robot_yaw])],
        )[0][:2]

    def _get_pose_in_frame(self, pose: Iterable, frame: Iterable):
        """
        pose    : vec2 in the format [position : vec3, orientation: vec4]
        frame   : vec2 in the format [position : vec3, orientation: vec4]
        """
        frame_inv = p.invertTransform(frame[0], frame[1])
        return p.multiplyTransforms(frame_inv[0], frame_inv[1], pose[0], pose[1])

    def _load_ground_plane(self):
        self.plane_height = 0  # -0.85

        path = os.path.join(get_urdf_path(), "plane/plane.urdf")
        self.plane_id = p.loadURDF(
            path,
            basePosition=[0, 0, self.plane_height],
            useFixedBase=True,
        )

    def _init_simulation(self):
        super()._init_simulation()

        # load plane
        self._load_ground_plane()
        # p.stepSimulation()

        # load goal
        fullpath = os.path.join(get_urdf_path(), "my_sphere.urdf")
        self.sphere = p.loadURDF(
            fullpath,
            basePosition=[0, 0, -5],  # hide it at first
            globalScaling=3,
            useFixedBase=True,
        )
        self.goal = np.zeros(2)
        # p.stepSimulation()

        self._load_robot()
        p.stepSimulation()

        # set gym spaces
        self.set_gym_spaces()

    def _seed(self, seed):
        for goal_space in self.goal_spaces:
            goal_space.seed(seed)
        self.action_space.seed(seed)
        self.should_reset = True

    def _reset(self, options={}):
        # Reset robot
        self._reset_robot(reset_position=self.should_reset)
        self.previous_action = np.zeros(self.n_actions)

        self.total_traveled = 0.0
        if not self.should_reset_robot_pos:
            mobile_base_state = p.getLinkState(
                self.robot_id, 0, computeLinkVelocity=False
            )
            self.pos_of_interest = np.array(mobile_base_state[0])
            self.initial_robot_pose = [mobile_base_state[0], mobile_base_state[1]]

        goal_space = self.goal_spaces[self.goal_space_size]
        if self.resample_goal or options.get("resample_goal", False):
            self.goal = self._sample_goal(
                goal_space, allow_out_of_bounds=options.get("allow_out_of_bounds", False)
            )

    def _reward(self) -> float:
        delta_pos = self._goal - self.pos_of_interest
        reward = -np.linalg.norm(delta_pos * self.error_bias)
        if np.linalg.norm(delta_pos) < self.distance_threshold:
            reward += self.reach_bonus
        return reward # type: ignore

    def _set_action(self, action):
        # action is the joint velocity
        action = np.clip(action, self.action_space.low, self.action_space.high) # type: ignore

        scaled_action = np.clip(
            action * self.robot_limits,
            -self.robot_limits,
            self.robot_limits,
        )
        # print(clipped_action)
        smoothed_action = self.limit_action(
            scaled_action,
            self.previous_action,
            self.robot_acceleration_limits,
            self.action_dt,
        )

        velocities = [smoothed_action[0]] * 4
        p.setJointMotorControlArray(
            self.robot_id,
            self.wheel_joint_ids,
            p.VELOCITY_CONTROL,
            targetVelocities=velocities,
            forces=[20, 20, 20, 20],
        )

        positions = [action[1], action[1], -action[1], -action[1]]
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

    def limit_action(self, action, prev_action, max_acceleration, dt) -> np.ndarray:
        # Calcul de la variation souhaitée
        delta = action - prev_action
        # Variation maximale autorisée durant dt
        max_delta = max_acceleration * dt
        # Limitation de la variation
        delta_limited = np.clip(delta, -max_delta, max_delta)
        # Commande finale
        return prev_action + delta_limited

    def _get_obs(self):
        wheel_states = p.getJointStates(
            self.robot_id,
            [
                self.robot_joint_info[joint][0]
                for joint in RR100ReachEnv.RR_VELOCITY_JOINTS
            ],
        )
        steering_states = p.getJointStates(
            self.robot_id,
            [
                i
                for i in [
                    self.robot_joint_info[joint][0]
                    for joint in RR100ReachEnv.RR_POSITION_JOINTS
                ]
            ],
        )
        mobile_base_state = p.getLinkState(self.robot_id, 0, computeLinkVelocity=True)
        self.robot_yaw = p.getEulerFromQuaternion(mobile_base_state[1])[2]
        self.pos_of_interest = np.array(mobile_base_state[0][:2])
        self.robot_velocity = np.array(mobile_base_state[6])
        yaw_velocity = mobile_base_state[7][2]

        rr100_wheel_velocities = np.array([wheel_states[0][1], wheel_states[1][1]])
        rr100_steering_angles = np.array([steering_states[0][0], steering_states[1][0]])
        rr100_steering_velocities = np.array(
            [steering_states[0][1], steering_states[1][1]]
        )

        mobile_base_angular = np.array([yaw_velocity])

        goal = self.goal
        if not self.should_reset_robot_pos and self.should_retransform_to_local:
            goal = self._get_pose_in_frame(
                [[*goal, 0.0], self.start_orientation],
                self.initial_robot_pose,
            )[0][:2]
            robot_pose_in_initial = self._get_pose_in_frame(
                [mobile_base_state[0], mobile_base_state[1]],
                self.initial_robot_pose,
            )
            robot_velocity_in_initial = self._get_pose_in_frame(
                [mobile_base_state[6], self.start_orientation],
                [[0, 0, 0], self.initial_robot_pose[1]],
            )
            robot_position = np.array(robot_pose_in_initial[0])
            robot_velocity = np.array(robot_velocity_in_initial[0])
            robot_yaw = p.getEulerFromQuaternion(robot_pose_in_initial[1])[2]
            mobile_base_orientation = np.array([robot_yaw])
        else:
            robot_position = self.pos_of_interest.copy()
            robot_velocity = self.robot_velocity.copy()
            mobile_base_orientation = np.array([self.robot_yaw])

        obs = np.concatenate(
            (
                goal,
                robot_position,
                rr100_wheel_velocities,
                rr100_steering_angles,
                rr100_steering_velocities,
                robot_velocity,
                mobile_base_orientation,
                mobile_base_angular,
            )
        ).astype(np.float32)

        return obs

    def _get_info(self):
        self.current_distance = np.linalg.norm(self._goal - self.pos_of_interest)
        info = {
            "goal": self._goal,
            "is_success": self._is_terminal(),
            "distance": self.current_distance,
            "initial_distance": self.initial_distance,
            "total_traveled": self.total_traveled,
        }
        return info

    def _sample_goal(self, goal_space, allow_out_of_bounds=False):
        goal = np.zeros(3)
        norm = np.linalg.norm(goal)
        tries = 0
        while norm < self.distance_threshold or (
            not allow_out_of_bounds and not goal in self.position_space
        ):
            goal = goal_space.sample()
            if not self.should_reset_robot_pos and self.should_retransform_to_local:
                goal_pose = p.multiplyTransforms(
                    self.initial_robot_pose[0],
                    self.initial_robot_pose[1],
                    [*goal, 0.0],
                    self.start_orientation,
                )
                goal = goal_pose[0]
                if tries >= 100:
                    self._reset_robot(reset_position=True)
                    tries = 0
            norm = np.linalg.norm(goal)
            tries += 1

        return goal

    def place_goal_debug(self, goal):
        p.resetBasePositionAndOrientation(
            self.sphere, [goal[0], goal[1], 0.75], self.start_orientation
        )

    def _is_terminal(self) -> bool:
        return self.current_distance < self.distance_threshold #type: ignore

    def _load_robot(self):
        path = os.path.join(get_urdf_path(), "RR100/rr100.urdf")

        self.robot_id = p.loadURDF(
            path,
            basePosition=[0, 0, 0],
            baseOrientation=self.start_orientation,
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_IMPLICIT_CYLINDER,
        )

        self.ur_rr100_num_joints = p.getNumJoints(self.robot_id)  # 26 joints
        joint_infos = [
            p.getJointInfo(self.robot_id, id) for id in range(self.ur_rr100_num_joints)
        ]
        self.robot_joint_info = {
            joint_info[1].decode(): joint_info[0:1] + joint_info[2:]
            for joint_info in joint_infos
        }

        for joint_name, joint_info in self.robot_joint_info.items():
            print(
                f"Joint: {joint_name}/{joint_info[0]}, joint type: {RR100ReachEnv.JOINT_TYPE_STR[joint_info[1]]}, limits: {(joint_info[7], joint_info[8])}"
            )
        self.wheel_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in RR100ReachEnv.RR_VELOCITY_JOINTS
        ]
        self.steering_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in RR100ReachEnv.RR_POSITION_JOINTS
        ]
        self.wheel_joint_ids = [
            self.robot_joint_info[joint][0]
            for joint in RR100ReachEnv.RR_VELOCITY_JOINTS
        ]
        self.steering_joint_ids = [
            self.robot_joint_info[joint][0]
            for joint in RR100ReachEnv.RR_POSITION_JOINTS
        ]

        self.wheel_velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in RR100ReachEnv.RR_VELOCITY_JOINTS
            ]
        )
        self.steering_position_limits = np.array(
            [
                (
                    self.robot_joint_info[joint][8 - 1],
                    self.robot_joint_info[joint][9 - 1],
                )  # lower, upper
                for joint in RR100ReachEnv.RR_POSITION_JOINTS
            ]
        )
        self.steering_velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in RR100ReachEnv.RR_POSITION_JOINTS
            ]
        )
        self.robot_limits = np.array(
            [
                np.pi,
                self.steering_position_limits[0][
                    0
                ],  # lower and upper limits are probably equal
            ]
        )
        self.robot_acceleration_limits = np.array(
            [
                self.wheel_acceleration_limit,
                self.steering_acceleration_limit,
            ]
        )
        self._set_initial_joints_positions()

    def _load_walls(self, min_x, max_x, min_y, max_y):
        path = os.path.join(
            get_urdf_path(),
            "my_cube.urdf",
        )
        x_size = abs(max_x - min_x)
        y_size = abs(max_y - min_y)
        self.wall_1 = p.loadURDF(
            path,
            basePosition=[max_x, 0, 0],
            baseOrientation=[0.0, 0.0, 0.707, 0.707],
            globalScaling=x_size,
            useFixedBase=True,
        )
        self.wall_2 = p.loadURDF(
            path,
            basePosition=[0, max_y, 0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            globalScaling=y_size,
            useFixedBase=True,
        )
        self.wall_3 = p.loadURDF(
            path,
            basePosition=[0, min_y, 0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            globalScaling=y_size,
            useFixedBase=True,
        )
        self.wall_4 = p.loadURDF(
            path,
            basePosition=[min_x, 0, 0],
            baseOrientation=[0.0, 0.0, 0.707, 0.707],
            globalScaling=x_size,
            useFixedBase=True,
        )

    def _set_initial_joints_positions(self):
        self.ur_joint_name_to_ids = {}
        # Set intial positions
        initial_positions = {
            "front_left_steering_joint": 0.0,
            "front_right_steering_joint": 0.0,
            "rear_left_steering_joint": 0.0,
            "rear_right_steering_joint": 0.0,
        }
        indices = (
            self.robot_joint_info[joint][0] for joint in initial_positions.keys()
        )

        for id, position in zip(indices, initial_positions.values()):
            p.resetJointState(self.robot_id, id, position)
        for joint in RR100ReachEnv.RR_VELOCITY_JOINTS:
            p.resetJointState(
                self.robot_id,
                self.robot_joint_info[joint][0],
                targetValue=0.0,
                targetVelocity=0.0,
            )

    def _reset_robot(self, reset_position):
        if reset_position:
            self._set_initial_joints_positions()
            p.resetBasePositionAndOrientation(
                self.robot_id, [0, 0, 0], self.start_orientation
            )
            self.initial_robot_pose = [np.zeros(3), self.start_orientation]
            self.pos_of_interest = np.zeros(2)
        p.stepSimulation()

    def _set_action_space(self):
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.n_actions,), dtype=np.float64
        )

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
                low=np.array([x_down, y_down]),
                high=np.array([x_up, y_up]),
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
                low=np.array([x_down, y_down]),
                high=np.array([x_up, y_up]),
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
                low=np.array([x_down, y_down]),
                high=np.array([x_up, y_up]),
                dtype=np.float64,
            )
        )

        if self.should_load_walls:
            self._load_walls(x_down, x_up, y_down, y_up)

        self.position_space = spaces.Box(
            low=np.array([x_down, y_down], dtype=np.float64),
            high=np.array([x_up, y_up], dtype=np.float64),
            dtype=np.float64,
        )

    def _set_observation_space(self):
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            shape=obs.shape,
            dtype=np.float32,
        )
        print("Observation space : ", self.observation_space)

    def debug_gui(self, low, high, color=[0, 0, 1]):
        low_array = low
        high_array = high

        p1 = [low_array[0], low_array[1], 0.9]  # xmin, ymin, zmin
        p2 = [high_array[0], low_array[1], 0.9]  # xmax, ymin, zmin
        p3 = [high_array[0], high_array[1], 0.9]  # xmax, ymax, zmin
        p4 = [low_array[0], high_array[1], 0.9]  # xmin, ymax, zmin

        p.addUserDebugLine(p1, p2, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p2, p3, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p3, p4, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        p.addUserDebugLine(p4, p1, lineColorRGB=color, lineWidth=2.0, lifeTime=0)

    @property
    def goal(self):
        return self._goal

    def manual_set_goal(
        self,
        goal: np.ndarray,
        retransform_to_local: bool = False,
        initial_pose: Optional[Iterable] = None,
        allow_out_of_bounds: bool = False,
    ) -> None:
        if not isinstance(goal, np.ndarray):
            goal = np.array(goal)
        assert goal.shape == (2,), "Goal shape error"
        assert allow_out_of_bounds or self.position_space.contains(
            goal
        ), "Goal out of bounds"

        if retransform_to_local:
            initial_pose = initial_pose or self.initial_robot_pose
            goal_pose = p.multiplyTransforms(
                initial_pose[0],
                initial_pose[1],
                [*goal, 0.0],
                self.start_orientation,
            )
            goal = goal_pose[0]

        self.goal = goal

    @goal.setter
    def goal(self, goal):
        self._goal = np.array(goal)
        self.initial_distance = np.linalg.norm(self._goal - self.pos_of_interest)
        self.place_goal_debug(goal)

    def _init_camera_matrices(self) -> None:
        camera_target_position = [0, 0, 0.5]
        camera_distance = 5
        camera_yaw = 90
        camera_pitch = -90
        camera_roll = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            camera_target_position,
            camera_distance,
            camera_yaw,
            camera_pitch,
            camera_roll,
            2,
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=40, aspect=1.7, nearVal=0.1, farVal=100
        )

    def _get_camera_projection_matrix(self):
        return self.projection_matrix

    def _get_camera_view_matrix(self):
        return self.view_matrix


class GoalPositionMixin(GoalEnvMixin):
    def _get_dict_obs(self):
        obs = self._get_obs()  # type: ignore
        return {
            "observation": obs,
            "achieved_goal": self.pos_of_interest.copy(),  # type: ignore
            "desired_goal": self.goal.copy(),  # type: ignore
        }

    def _set_observation_space(self):
        obs = self._get_dict_obs()
        obs_spaces = {
            "observation": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                shape=obs["observation"].shape,
                dtype=np.float32,
            ),
            "achieved_goal": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                shape=obs["achieved_goal"].shape,
                dtype=np.float32,
            ),
            "desired_goal": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                shape=obs["desired_goal"].shape,
                dtype=np.float32,
            ),
        }
        self.observation_space = spaces.Dict(obs_spaces)  # type: ignore

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Union[dict, list, np.ndarray],
    ) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if isinstance(info, dict):
            info["distance"] = distance
        elif isinstance(info, list) or isinstance(info, np.ndarray):
            for i in range(len(info)):
                info[i]["distance"] = distance[i]
        return -(distance > self.distance_threshold).astype(np.float32)  # type: ignore


class GoalPoseMixin(GoalEnvMixin):
    def _get_dict_obs(self):
        obs = self._get_obs()  # type: ignore
        return {
            "observation": obs,
            "achieved_goal": self.pose_of_interest.copy(),  # type: ignore
            "desired_goal": self.goal.copy(),  # type: ignore
        }

    def _set_observation_space(self):
        obs = self._get_dict_obs()
        obs_spaces = {
            "observation": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                shape=obs["observation"].shape,
                dtype=np.float32,
            ),
            "achieved_goal": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                shape=obs["achieved_goal"].shape,
                dtype=np.float32,
            ),
            "desired_goal": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                shape=obs["desired_goal"].shape,
                dtype=np.float32,
            ),
        }
        self.observation_space = spaces.Dict(obs_spaces)  # type: ignore

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Union[dict, list, np.ndarray],
    ) -> float:
        distance = np.linalg.norm(achieved_goal[:-1] - desired_goal[:-1], axis=-1)
        angle_diff = wrapped_angle_difference(achieved_goal[-1], desired_goal[-1])

        if isinstance(info, dict):
            info["distance"] = distance
            info["angle_diff"] = angle_diff
        elif isinstance(info, list) or isinstance(info, np.ndarray):
            for i in range(len(info)):
                info[i]["distance"] = distance[i]
                info[i]["angle_diff"] = distance[i]
        return -(distance > self.distance_threshold and angle_diff > self.angle_threshold).astype(np.float32)  # type: ignore


class RR100ReachGoalEnv(GoalPositionMixin, RR100ReachEnv):
    pass


if __name__ == "__main__":
    env = RR100ReachEnv()
