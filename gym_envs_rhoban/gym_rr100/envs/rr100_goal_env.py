from abc import abstractmethod
from enum import IntEnum
import math
import os
from typing import Any, Iterable, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data as pd

from gym_envs_rhoban import get_urdf_path


class GoalSpaceSize(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


class GoalEnv(gym.Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class RR100ReachGoalEnv(GoalEnv):

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    RR_VELOCITY_JOINTS: list[str] = [
        "front_left_wheel",
        "front_right_wheel",
        "rear_left_wheel",
        "rear_right_wheel",
    ]
    RR_POSITION_JOINTS: list[str] = [
        "front_left_steering_joint",
        "front_right_steering_joint",
        "rear_left_steering_joint",
        "rear_right_steering_joint",
    ]

    def __init__(
        self,
        render_mode="human",
        goal_space_size: GoalSpaceSize = GoalSpaceSize.SMALL,
        distance_threshold: float = 0.10,
        robot_action_frequency: int = 40,
        wheel_acceleration_limit: float = 2 * np.pi,
        steering_acceleration_limit: float = np.pi / 6,
        should_load_walls: bool = True,
        is_rl_ackermann: bool = False,
        should_reset_robot_position: bool = True,
        should_retransform_to_local: bool = False,
    ):

        self.is_rl_ackermann = is_rl_ackermann
        self.should_reset_robot_pos = should_reset_robot_position
        self.should_retransform_to_local = should_retransform_to_local
        self.total_episodes = 0
        self.initial_distance = 0.0
        self.total_traveled = 0.0
        self.robot_position = np.zeros(2)
        self.initial_robot_pose = [np.zeros(3), np.zeros(4)]

        # switch tool to convert a numeric value to string value
        self.switcher_type_name = {
            p.JOINT_REVOLUTE: "JOINT_REVOLUTE",
            p.JOINT_PRISMATIC: "JOINT_PRISMATIC",
            p.JOINT_SPHERICAL: "JOINT SPHERICAL",
            p.JOINT_PLANAR: "JOINT PLANAR",
            p.JOINT_FIXED: "JOINT FIXED",
        }

        self.goal_space_size = goal_space_size
        self.should_load_walls = should_load_walls

        # bullet parameters
        self.time_step = 1.0 / 240
        self.n_substeps = 10

        self.robot_action_frequency = robot_action_frequency  # Hz
        self.action_dt = 1 / self.robot_action_frequency  # s

        self.wheel_acceleration_limit = wheel_acceleration_limit
        self.steering_acceleration_limit = steering_acceleration_limit

        # TEMPORARY, should probably read from URDF or pass as argument
        self.steering_track = 0.6
        self.wheel_base = 0.5 / 2
        self.wheel_radius = 0.21
        self.steering_position_limit = 0.347

        # robot parameters
        self.distance_threshold = distance_threshold
        self.rr100_base_index = 0
        self.max_vel = 1
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # connect bullet
        simulation_type = p.DIRECT if render_mode != "human" else p.GUI
        p.connect(
            simulation_type
        )  # or p.GUI (for test) or p.DIRECT (for train) for non-graphical version

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
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
        self.width, self.height = 1920, 1200

        self.if_render = False
        self.render_mode = render_mode

        self._init_simulation()
        self.previous_action = np.zeros(self.n_actions)

        self.debug_gui(self.position_space.low, self.position_space.high, [0, 0, 1])
        self.debug_gui(
            self.goal_spaces[GoalSpaceSize.SMALL].low,
            self.goal_spaces[GoalSpaceSize.SMALL].high,
            [1, 0, 0],
        )
        self.debug_gui(
            self.goal_spaces[goal_space_size].low,
            self.goal_spaces[goal_space_size].high,
            [1, 0, 1],
        )

        # gym setup
        # self.goal = self._sample_goal()

        p.stepSimulation()

    def printAllInfo(self):
        print("=================================")
        print("All Robot joints info")
        num_joints = p.getNumJoints(self.robot_id)
        print("=> num of joints = {0}".format(num_joints))
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            # print(joint_info)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]
            child_link_name = joint_info[12].decode("UTF-8")
            link_pos_in_parent_frame = p.getLinkState(self.robot_id, i)[0]
            link_orien_in_parent_frame = p.getLinkState(self.robot_id, i)[1]
            joint_type_name = self.switcher_type_name.get(joint_type, "Invalid type")
            joint_lower_limit, joint_upper_limit = joint_info[8:10]
            joint_limit_effort = joint_info[10]
            joint_limit_velocity = joint_info[11]
            print(
                "i={0}, name={1}, type={2}, lower={3}, upper={4}, effort={5}, velocity={6}".format(
                    i,
                    joint_name,
                    joint_type_name,
                    joint_lower_limit,
                    joint_upper_limit,
                    joint_limit_effort,
                    joint_limit_velocity,
                )
            )
            print(
                "child link name={0}, pos={1}, orien={2}".format(
                    child_link_name,
                    link_pos_in_parent_frame,
                    link_orien_in_parent_frame,
                )
            )
        print("=================================")

    def _get_position_in_robot_frame(self, position: np.ndarray):
        robot_pos = np.concatenate((self.robot_position, [0.0]))
        return self._get_pose_in_frame(
            [position, self.start_orientation],
            [robot_pos, p.getQuaternionFromEuler([0, 0, self.robot_yaw])],
        )[0]

    def _get_pose_in_frame(self, pose: Iterable, frame: Iterable):
        """
        pose    : vec2 in the format [position : vec3, orientation: vec4]
        frame   : vec2 in the format [position : vec3, orientation: vec4]
        """
        frame_inv = p.invertTransform(frame[0], frame[1])
        return p.multiplyTransforms(frame_inv[0], frame_inv[1], pose[0], pose[1])

    def load_plane(self):
        self.plane_height = 0  # -0.85

        path = os.path.join(get_urdf_path(), "plane/plane.urdf")
        self.plane_id = p.loadURDF(
            path,
            basePosition=[0, 0, self.plane_height],
            useFixedBase=True,
        )
        # print(p.getDynamicsInfo(self.plane_id, -1))

    def _init_simulation(self):
        p.resetSimulation()

        # bullet setup
        # self.seed()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(numSubSteps=self.n_substeps)

        p.setGravity(0, 0, -9.81)

        # load plane
        self.load_plane()
        p.stepSimulation()

        # load goal
        fullpath = os.path.join(get_urdf_path(), "my_sphere.urdf")
        self.sphere = p.loadURDF(fullpath, useFixedBase=True)
        p.stepSimulation()
        self.goal = [0, 0]

        # load panda
        self.load_robot()
        p.stepSimulation()

        self.set_rr100_initial_joints_positions()
        p.stepSimulation()

        # set gym spaces
        self.set_gym_spaces_rr100()

        p.stepSimulation()

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
        return -(distance > self.distance_threshold).astype(np.float32)

    # basic methods
    # -------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        P_1 = self.robot_position.copy()

        # for i in range(240):
        for _ in range(int((1 / self.time_step) // self.robot_action_frequency)):
            p.stepSimulation()

        obs = self._get_obs()

        self.total_traveled += np.linalg.norm(self.robot_position - P_1)

        info = {}
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        info["goal"] = self.goal
        info["initial_distance"] = self.initial_distance
        info["total_traveled"] = self.total_traveled
        info["is_success"] = info["distance"] < self.distance_threshold

        terminated = info["is_success"]
        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed, options=options)
        if seed is not None:
            for goal_space in self.goal_spaces:
                goal_space.seed(seed)
            self.action_space.seed(seed)

        # Reset robot
        self.reset_robot(self.should_reset_robot_pos)
        self.previous_action = np.zeros(self.n_actions)

        self.total_traveled = 0.0
        if not self.should_reset_robot_pos:
            mobile_base_state = p.getLinkState(
                self.robot_id, 0, computeLinkVelocity=False
            )
            self.robot_position = np.array(mobile_base_state[0])[:2]
            self.initial_robot_pose = [mobile_base_state[0], mobile_base_state[1]]

        goal_space = self.goal_spaces[self.goal_space_size]
        self.goal = self._sample_goal(goal_space)

        obs = self._get_obs()
        info = {}
        _ = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        info["is_success"] = info["distance"] < self.distance_threshold

        self.total_episodes += 1

        return obs, info

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            return None

        if self.render_mode == "rgb_array":
            return p.getCameraImage(
                self.width,
                self.height,
                self.view_matrix,
                self.projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )[2]

    def reward(self):
        # Get goal position in robot's frame
        goal = np.array([*self.goal, 0.0])
        self.goal_robot_frame = self._get_position_in_robot_frame(goal)[:2]

        reward = -np.linalg.norm(self.goal_robot_frame * self.error_bias)

        return reward, np.linalg.norm(self.goal_robot_frame)

    # RobotEnv method
    # -------------------------

    def _set_action(self, action):
        # action is the joint velocity
        # print("before step :", p.getJointState(self.panda_id, self.arm_eef_index)[0])
        assert action.shape == (self.n_actions,), "Action shape error"

        if self.is_rl_ackermann:
            self._rl_ackermann_set_action(action)
        else:
            self._simple_set_action(action)

    def _simple_set_action(self, action):
        # Use sign of left wheels for both sides to avoid not moving
        action[1] = np.sign(action[0]) * abs(action[1])
        action[3] = np.sign(action[2]) * abs(action[3])

        clipped_action = np.clip(
            action * self.robot_velocity_limits,
            -self.robot_velocity_limits,
            self.robot_velocity_limits,
        )
        # print(clipped_action)
        smoothed_action = self.limit_action(
            clipped_action,
            self.previous_action,
            self.robot_acceleration_limits,
            self.action_dt,
        )

        velocities = [
            smoothed_action[0],
            smoothed_action[1],
            smoothed_action[0],
            smoothed_action[1],
        ]
        # print(f"Wheel velocities : {velocities}")
        p.setJointMotorControlArray(
            self.robot_id,
            self.wheel_joint_ids,
            p.VELOCITY_CONTROL,
            targetVelocities=velocities,
            forces=[20, 20, 20, 20],
        )

        positions = [action[2], action[3], -action[2], -action[3]]
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
        # print("Applied action : ", velocities + positions)

    def _rl_ackermann_set_action(self, action):
        clipped_action = np.clip(
            action * self.robot_velocity_limits,
            -self.robot_velocity_limits,
            self.robot_velocity_limits,
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

    def limit_action(self, action, prev_action, max_acceleration, dt) -> np.ndarray:
        """
        Limite la variation entre current_cmd et previous_cmd.
        :param current_cmd: Commande souhaitée actuelle (par exemple, vitesse ou angle).
        :param previous_cmd: Commande appliquée lors du pas précédent.
        :param max_acceleration: Variation maximale autorisée par seconde.
        :param dt: Intervalle de temps entre deux pas de simulation.
        :return: Commande lissée à appliquer.
        """
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
                for joint in RR100ReachGoalEnv.RR_VELOCITY_JOINTS
            ],
        )
        steering_states = p.getJointStates(
            self.robot_id,
            [
                i
                for i in [
                    self.robot_joint_info[joint][0]
                    for joint in RR100ReachGoalEnv.RR_POSITION_JOINTS
                ]
            ],
        )
        mobile_base_state = p.getLinkState(self.robot_id, 0, computeLinkVelocity=True)
        self.robot_yaw = p.getEulerFromQuaternion(mobile_base_state[1])[2]
        self.robot_position = np.array(mobile_base_state[0])[:2]
        self.robot_velocity = np.array(mobile_base_state[6])[:2]
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
                [np.concatenate((goal, [0.0])), self.start_orientation],
                self.initial_robot_pose,
            )[0]
            robot_pose_in_initial = self._get_pose_in_frame(
                [mobile_base_state[0], mobile_base_state[1]],
                self.initial_robot_pose,
            )
            robot_velocity_in_initial = self._get_pose_in_frame(
                [mobile_base_state[6], self.start_orientation],
                [[0, 0, 0], self.initial_robot_pose[1]],
            )
            robot_position = np.array(robot_pose_in_initial[0][:2])
            robot_velocity = np.array(robot_velocity_in_initial[0][:2])
            robot_yaw = p.getEulerFromQuaternion(robot_pose_in_initial[1])[2]
            mobile_base_orientation = np.array([robot_yaw])
        else:
            robot_position = self.robot_position.copy()
            robot_velocity = self.robot_velocity.copy()
            mobile_base_orientation = np.array([self.robot_yaw])

        obs = np.concatenate(
            (
                robot_position,
                goal[:2],
                rr100_wheel_velocities,
                rr100_steering_angles,
                rr100_steering_velocities,
                robot_velocity,
                mobile_base_orientation,
                mobile_base_angular,
            )
        ).astype(np.float32)

        return {
            "observation": obs,
            "achieved_goal": robot_position,
            "desired_goal": np.array(goal[:2]),
        }

    def _get_info(self, distance):
        info = {
            "goal": self.goal,
            "is_success": self._is_success(distance),
            "distance": distance,
            "initial_distance": self.initial_distance,
            "total_traveled": self.total_traveled,
        }
        return info

    def _sample_goal(self, goal_space):
        goal = np.zeros(2)
        norm = np.linalg.norm(goal)
        tries = 0
        while norm < self.distance_threshold or not goal in self.position_space:
            goal = goal_space.sample()
            if not self.should_reset_robot_pos and self.should_retransform_to_local:
                goal_pose = p.multiplyTransforms(
                    self.initial_robot_pose[0],
                    self.initial_robot_pose[1],
                    [*goal, 0.0],
                    self.start_orientation,
                )
                goal = goal_pose[0][:2]
                if tries >= 100:
                    self.reset_robot(reset_position=True)
                    tries = 0
            norm = np.linalg.norm(goal)
            tries += 1

        return goal

    def place_goal_debug(self, goal):
        p.resetBasePositionAndOrientation(
            self.sphere, [goal[0], goal[1], 0.75], self.start_orientation
        )

    def _is_success(self, distance_error):
        return distance_error < self.distance_threshold

    def load_robot(self):
        self.ur_rr100_start_pos = [0, 0, 0]
        self.ur_rr100_start_orientation = self.start_orientation

        path = os.path.join(get_urdf_path(), "RR100/rr100.urdf")

        self.robot_id = p.loadURDF(
            path,
            basePosition=self.ur_rr100_start_pos,
            baseOrientation=self.ur_rr100_start_orientation,
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
        self.wheel_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in RR100ReachGoalEnv.RR_VELOCITY_JOINTS
        ]
        self.steering_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in RR100ReachGoalEnv.RR_POSITION_JOINTS
        ]
        self.wheel_joint_ids = [
            self.robot_joint_info[joint][0]
            for joint in RR100ReachGoalEnv.RR_VELOCITY_JOINTS
        ]
        self.steering_joint_ids = [
            self.robot_joint_info[joint][0]
            for joint in RR100ReachGoalEnv.RR_POSITION_JOINTS
        ]

        self.velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in RR100ReachGoalEnv.RR_VELOCITY_JOINTS
            ]
        )
        self.position_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in RR100ReachGoalEnv.RR_POSITION_JOINTS
            ]
        )
        self.steering_velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in RR100ReachGoalEnv.RR_POSITION_JOINTS
            ]
        )
        print("Robot dynamics : ", p.getDynamicsInfo(self.robot_id, -1))
        if self.is_rl_ackermann:
            self.robot_velocity_limits = np.array(
                [
                    np.pi,
                    self.steering_position_limit,
                ]
            )
            self.robot_acceleration_limits = np.array(
                [
                    self.wheel_acceleration_limit,
                    self.steering_acceleration_limit,
                ]
            )
        else:
            self.robot_velocity_limits = np.array(
                [
                    np.pi,
                    np.pi,
                    self.position_limits[0],
                    self.position_limits[0],
                ]
            )
            self.robot_acceleration_limits = np.array(
                [
                    self.wheel_acceleration_limit,
                    self.wheel_acceleration_limit,
                    self.steering_acceleration_limit,
                    self.steering_acceleration_limit,
                ]
            )

    def load_walls(self, min_x, max_x, min_y, max_y):
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

    def set_rr100_initial_joints_positions(self):
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
        for joint in RR100ReachGoalEnv.RR_VELOCITY_JOINTS:
            p.resetJointState(
                self.robot_id,
                self.robot_joint_info[joint][0],
                targetValue=0.0,
                targetVelocity=0.0,
            )

    def reset_robot(self, reset_position):
        self.set_rr100_initial_joints_positions()
        if reset_position:
            p.resetBasePositionAndOrientation(
                self.robot_id, [0, 0, 0], self.start_orientation
            )
            self.robot_position = np.zeros(2)
        p.stepSimulation()

    def set_gym_spaces_rr100(self):
        """# Min/Max Z
        z_down = 0.64
        z_up = 1.34"""

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

        self.position_space = spaces.Box(
            low=np.array([x_down, y_down], dtype=np.float64),
            high=np.array([x_up, y_up], dtype=np.float64),
            dtype=np.float64,
        )

        if self.should_load_walls:
            self.load_walls(x_down, x_up, y_down, y_up)

        # action_space = joint position (6 float) + mobile base velocity and steering angle (2 float) = 8 float
        self.n_actions = 2 if self.is_rl_ackermann else 4
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.n_actions,), dtype=np.float64
        )

        obs = self._get_obs()
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

        self.observation_space = spaces.Dict(obs_spaces)

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

    @goal.setter
    def goal(self, goal):
        self._goal = np.array(goal)
        self.initial_distance = np.linalg.norm(self._goal - self.robot_position)
        self.place_goal_debug(goal)

    def close(self):
        p.disconnect()
