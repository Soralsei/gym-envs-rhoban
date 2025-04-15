from enum import IntEnum
import math
import os
from typing import Any, Iterable
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


class RR100ReachEnv(gym.Env):

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
        error_bias: Iterable[float] = np.array([1.0, 1.0]),
        should_load_walls: bool = True,
        is_rl_ackermann: bool = False,
    ):

        self.is_rl_ackermann = is_rl_ackermann

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

        # error bias
        if not isinstance(error_bias, np.ndarray):
            error_bias = np.array(error_bias)
        self.error_bias = error_bias

        # connect bullet
        simulation_type = p.DIRECT if render_mode != "human" else p.GUI
        p.connect(
            simulation_type
        )  # or p.GUI (for test) or p.DIRECT (for train) for non-graphical version

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        self.if_render = False
        self.render_mode = render_mode

        self.goal = [0, 0]
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

        self.total_episodes = 0

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

    def get_pose_in_robot_frame(self, pose: np.ndarray):
        robot_pos = np.concatenate((self.robot_position, [0.0]))
        robot_inv = p.invertTransform(
            robot_pos, p.getQuaternionFromEuler([0, 0, self.robot_yaw])
        )
        pose_in_robot_frame = np.array(
            p.multiplyTransforms(
                robot_inv[0], robot_inv[1], pose, self.start_orientation
            )[0]
        )
        return pose_in_robot_frame

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

        # load panda
        self.load_robot()
        p.stepSimulation()

        self.set_rr100_initial_joints_positions()
        p.stepSimulation()

        # set gym spaces
        self.set_gym_spaces_rr100()

        p.stepSimulation()

    # basic methods
    # -------------------------
    def step(self, action):
        # print(f'Action : {action}')
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        # for i in range(240):
        for _ in range(int((1 / self.time_step) // self.robot_action_frequency)):
            p.stepSimulation()

        obs = self._get_obs()

        reward, distance = self.reward()
        info = self._get_info(distance)

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
        self.reset_robot()
        self.previous_action = np.zeros(self.n_actions)

        goal_space = self.goal_spaces[self.goal_space_size]
        self.goal = self._sample_goal(goal_space)

        obs = self._get_obs()
        _, distance = self.reward()
        info = self._get_info(distance)

        self.total_episodes += 1

        return obs, info

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            return None

        if self.render_mode == "rgb_array":
            # (width, height, rgbPixels, depthPixels, segmentationMaskBuffer)
            return p.getCameraImage(640, 360)[2]

    def reward(self):
        # Get goal position in robot's frame
        goal = np.array([self.goal[0], self.goal[1], 0.0])
        self.goal_robot_frame = self.get_pose_in_robot_frame(goal)[:2]

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
            )
        )
        phi_o = math.atan2(
            (2 * self.wheel_base * math.sin(smoothed_action[1])),
            (
                2 * self.wheel_base * math.cos(smoothed_action[1])
                + side * self.steering_track * math.sin(smoothed_action[1])
            )
        )
        R_steering_i = self.wheel_base * math.tan(np.pi/2 - abs(phi_i) + 1e-9)
        R_steering_o = R_steering_i + self.steering_track
        R_steering_c = R_steering_i + (self.steering_track / 2)

        R_w_i = np.sqrt(R_steering_i**2 + self.wheel_base**2)
        R_w_o = np.sqrt(R_steering_o**2 + self.wheel_base**2)
        R_w_c = np.sqrt(R_steering_c**2 + self.wheel_base**2)

        w_i = smoothed_action[0] * R_w_i / R_w_c
        w_o = smoothed_action[0] * R_w_o / R_w_c

        # print(f"phi_i = {phi_i}")
        # print(f"phi_o = {phi_o}")
        # print(f"R_steering_c = {R_steering_c}")
        # print(f"R_steering_i = {R_steering_i}")
        # print(f"R_steering_o = {R_steering_o}")
        # print(f"R_w_c = {R_w_c}")
        # print(f"R_w_i = {R_w_i}")
        # print(f"R_w_o = {R_w_o}")
        # print(f"w_c = {smoothed_action[0]}")
        # print(f"w_i = {w_i}")
        # print(f"w_o = {w_o}")

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
        self.robot_position = np.array(mobile_base_state[0])[:2]
        self.robot_velocity = np.array(mobile_base_state[6])[:2]
        yaw_velocity = mobile_base_state[7][2]

        rr100_wheel_velocities = np.array([wheel_states[0][1], wheel_states[1][1]])
        rr100_steering_angles = np.array([steering_states[0][0], steering_states[1][0]])
        rr100_steering_velocities = np.array(
            [steering_states[0][1], steering_states[1][1]]
        )

        mobile_base_orientation = np.array([self.robot_yaw])
        mobile_base_angular = np.array([yaw_velocity])

        obs = np.concatenate(
            (
                self.robot_position.copy(),
                self.goal,
                rr100_wheel_velocities,
                rr100_steering_angles,
                rr100_steering_velocities,
                self.robot_velocity.copy(),
                mobile_base_orientation,
                mobile_base_angular,
            )
        ).astype(np.float32)

        return obs

    def _get_info(self, distance):
        info = {
            "is_success": self._is_success(distance),
            "distance": distance,
        }
        return info

    def _sample_goal(self, goal_space):
        goal = np.zeros(2)
        while np.linalg.norm(goal) < self.distance_threshold:
            goal = goal_space.sample()

        p.resetBasePositionAndOrientation(
            self.sphere, [goal[0], goal[1], 0.9], self.start_orientation
        )
        return goal.copy()

    def _is_success(self, distance_error):
        return distance_error < self.distance_threshold

    def load_robot(self):
        self.ur_rr100_start_pos = [0, 0, 0]
        self.ur_rr100_start_orientation = self.start_orientation

        path = os.path.join(get_urdf_path(), "RR100/rr100_ur.urdf")

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

        self.velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in RR100ReachEnv.RR_VELOCITY_JOINTS
            ]
        )
        self.position_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in RR100ReachEnv.RR_POSITION_JOINTS
            ]
        )
        self.steering_velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in RR100ReachEnv.RR_POSITION_JOINTS
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

        # for joint in RR100ReachEnv.RR_VELOCITY_JOINTS:
        #     joint_info = p.getJointInfo(self.robot_id, self.robot_joint_info[joint][0])
        #     print(f"Wheel info : {joint_info}")
        # for joint in RR100ReachEnv.RR_POSITION_JOINTS:
        #     joint_info = p.getJointInfo(self.robot_id, self.robot_joint_info[joint][0])
        #     print(f"Wheel info : {joint_info}")
        # p.changeDynamics(
        #     self.robot_id,
        #     self.robot_joint_info[joint][0],
        #     maxJointVelocity=self.robot_joint_info[joint][11 - 1],
        # )

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
        for joint in RR100ReachEnv.RR_VELOCITY_JOINTS:
            p.resetJointState(
                self.robot_id,
                self.robot_joint_info[joint][0],
                targetValue=0.0,
                targetVelocity=0.0,
            )

    def reset_robot(self):
        self.set_rr100_initial_joints_positions()
        p.resetBasePositionAndOrientation(
            self.robot_id, [0, 0, 0], self.start_orientation
        )
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
        self.observation_space = spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            shape=obs.shape,
            # shape=(num_wheels + num_joints_steering * 2 + 3 + 3 + 3 + 3 + 3,),
            dtype=np.float32,
        )
        print("Observation space : ", self.observation_space)

    def debug_gui(self, low, high, color=[0, 0, 1]):
        # low_array = self.pos_space.low
        # high_array = self.pos_space.high

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

    def close(self):
        p.disconnect()
