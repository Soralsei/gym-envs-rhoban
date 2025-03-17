from enum import IntEnum
import os
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data as pd
from itertools import repeat

from gym_envs import get_urdf_path


class GoalSpaceSize(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


class UrRrReachEnv(gym.Env):

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    UR_CONTROLLABLE_JOINTS: list[str] = [
        "ur_shoulder_pan_joint",
        "ur_shoulder_lift_joint",
        "ur_elbow_joint",
        "ur_wrist_1_joint",
        "ur_wrist_2_joint",
        # "ur_wrist_3_joint",
    ]
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
        distance_threshold: float = 0.05,
        robot_action_frequency: int = 40,
        ur_joint_acceleration_limit: float = 2 * np.pi,
        wheel_acceleration_limit: float = 2 * np.pi,
        steering_acceleration_limit: float = np.pi / 6,
    ):

        self.use_ur_only = False
        self.use_ackermann = False

        # switch tool to convert a numeric value to string value
        self.switcher_type_name = {
            p.JOINT_REVOLUTE: "JOINT_REVOLUTE",
            p.JOINT_PRISMATIC: "JOINT_PRISMATIC",
            p.JOINT_SPHERICAL: "JOINT SPHERICAL",
            p.JOINT_PLANAR: "JOINT PLANAR",
            p.JOINT_FIXED: "JOINT FIXED",
        }

        self.goal_space_size = goal_space_size

        # bullet parameters
        self.time_step = 1.0 / 240
        self.n_substeps = 10

        self.robot_action_frequency = robot_action_frequency  # Hz
        self.action_dt = 1 / self.robot_action_frequency  # s

        self.ur_joint_acceleration_limit = ur_joint_acceleration_limit
        self.wheel_acceleration_limit = wheel_acceleration_limit
        self.steering_acceleration_limit = steering_acceleration_limit
        self.rr_acceleration_limits = np.array(
            [self.wheel_acceleration_limit, self.steering_acceleration_limit]
        )

        # robot parameters
        self.distance_threshold = distance_threshold
        self.arm_eef_index = 24
        self.max_vel = 1
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # connect bullet
        simulation_type = p.DIRECT if render_mode != "human" else p.GUI
        p.connect(
            simulation_type
        )  # or p.GUI (for test) or p.DIRECT (for train) for non-graphical version
        self.if_render = False
        self.render_mode = render_mode

        self.goal = np.array([0, 0, 0])
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

    def get_pose_in_local_frame(
        self, pose: np.ndarray, other_position, other_orientation
    ):
        robot_inv = p.invertTransform(other_position, other_orientation)
        pose_in_local_frame = np.array(
            p.multiplyTransforms(
                robot_inv[0], robot_inv[1], pose, self.start_orientation
            )[0]
        )
        return pose_in_local_frame

    def load_plane(self):
        self.plane_height = 0  # -0.85

        path = os.path.join(get_urdf_path(), "plane/plane.urdf")
        self.plane_id = p.loadURDF(
            path,
            basePosition=[0, 0, self.plane_height],
            useFixedBase=True,
        )

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

        # set panda joints to initial positions
        self.set_ur_initial_joints_positions()
        p.stepSimulation()

        # set gym spaces
        if self.use_ur_only:
            self.set_gym_spaces_ur()
        else:
            self.set_gym_spaces_ur_rr100()

        p.stepSimulation()

    # basic methods
    # -------------------------
    def step(self, action):
        # print(f'Action : {action}')
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # print(action)
        self._set_action(action)

        # for i in range(240):
        for _ in range(int((1 / self.time_step) // self.robot_action_frequency)):
            p.stepSimulation()

        obs = self._get_obs()

        reward, distance = self.reward()
        info = self._get_info(reward, distance)

        terminated = info["is_success"]
        truncated = False

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enable if want to control rendering
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

        # set panda joints to initial positions
        self.set_ur_initial_joints_positions()
        p.resetBasePositionAndOrientation(
            self.robot_id, [0, 0, 0], self.start_orientation
        )
        p.stepSimulation()

        goal_space = self.goal_spaces[self.goal_space_size]
        self.goal = self._sample_goal(goal_space)

        obs = self._get_obs()
        reward, distance = self.reward()
        info = self._get_info(reward, distance)

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
        '''
        Simple euclidean distance between gripper and goal
        '''
        delta_pos = self.goal - self.gripper_position
        dist = np.linalg.norm(delta_pos)

        return -dist, dist

    # RobotEnv method
    # -------------------------

    def _set_action(self, action):
        # action is the joint velocity
        # print("before step :", p.getJointState(self.panda_id, self.arm_eef_index)[0])
        num_joints_ur = len(UrRrReachEnv.UR_CONTROLLABLE_JOINTS)

        if self.use_ur_only:
            assert action.shape == (num_joints_ur,), "Action shape error"

        else:
            assert action.shape == (self.n_actions,), "Action shape error"

        clipped_action_ur = np.clip(
            action[:num_joints_ur] * self.velocity_limits[:num_joints_ur],
            -self.velocity_limits[:num_joints_ur],
            self.velocity_limits[:num_joints_ur],
        )
        smoothed_ur_action = self.post_process_action(
            clipped_action_ur,
            self.previous_action[:num_joints_ur],
            self.ur_joint_acceleration_limit,
            self.action_dt,
        )
        effective_action = [smoothed_ur_action]

        p.setJointMotorControlArray(
            self.robot_id,
            self.ur_joint_ids,
            p.VELOCITY_CONTROL,
            targetVelocities=smoothed_ur_action,
            forces=self.ur_joint_forces,
        )

        if not self.use_ur_only:
            clipped_action_rr100 = np.clip(
                action[num_joints_ur:] * self.rr100_velocity_steering_limits,
                -self.rr100_velocity_steering_limits,
                self.rr100_velocity_steering_limits,
            )
            # print(
            #     f"Clipped action (after scaling with limits) : {clipped_action_rr100}"
            # )
            smoothed_action = self.post_process_action(
                clipped_action_rr100,
                self.previous_action[num_joints_ur:],
                self.rr_acceleration_limits,
                self.action_dt,
            )
            # print(f"Acceleration- limited action : {smoothed_action}")

            velocities = [smoothed_action[0]] * len(self.wheel_joint_ids)
            p.setJointMotorControlArray(
                self.robot_id,
                self.wheel_joint_ids,
                p.VELOCITY_CONTROL,
                targetVelocities=velocities,
                forces=[10, 10, 10, 10],
            )

            signs = np.array(
                [
                    1 if "front" in name else -1
                    for name in UrRrReachEnv.RR_POSITION_JOINTS
                ]
            )
            positions = signs * smoothed_action[1]
            p.setJointMotorControlArray(
                self.robot_id,
                self.steering_joint_ids,
                p.POSITION_CONTROL,
                targetPositions=positions,
                forces=self.steering_joint_forces,
            )

            effective_action.append(smoothed_action)

        elif self.use_ackermann:
            pass

            """for i in range(2):
                # Appliquer les angles de braquage
                p.setJointMotorControl2(robot, FRONT_LEFT_STEER, p.POSITION_CONTROL, targetPosition=theta_FL)
                p.setJointMotorControl2(robot, FRONT_RIGHT_STEER, p.POSITION_CONTROL, targetPosition=theta_FR)
                p.setJointMotorControl2(robot, REAR_LEFT_STEER, p.POSITION_CONTROL, targetPosition=theta_RL)
                p.setJointMotorControl2(robot, REAR_RIGHT_STEER, p.POSITION_CONTROL, targetPosition=theta_RR)

                # Appliquer la vitesse aux roues motrices
                p.setJointMotorControl2(robot, FRONT_LEFT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=speed)
                p.setJointMotorControl2(robot, FRONT_RIGHT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=speed)
                p.setJointMotorControl2(robot, REAR_LEFT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=speed)
                p.setJointMotorControl2(robot, REAR_RIGHT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=speed)"""

        self.previous_action = np.concatenate(effective_action)
        # print("Applied action : ", self.previous_action)

    def post_process_action(
        self, action, prev_action, max_acceleration, dt
    ) -> np.ndarray:
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
        ur_joint_states = p.getJointStates(
            self.robot_id,
            [
                self.robot_joint_info[joint][0]
                for joint in UrRrReachEnv.UR_CONTROLLABLE_JOINTS
            ],
        )
        ur_joint_positions = np.array([state[0] for state in ur_joint_states])
        ur_joint_velocities = np.array([state[1] for state in ur_joint_states])
        ee_state = p.getLinkState(
            self.robot_id, self.arm_eef_index, computeLinkVelocity=1
        )
        self.gripper_position = np.array(ee_state[0])
        self.gripper_orientation = ee_state[1]
        self.gripper_velocity = np.array(ee_state[6])
        ur_base_frame = p.getLinkState(self.robot_id, self.ur_base_link_id)
        translate_inv, orient_inv = p.invertTransform(ur_base_frame[0], ur_base_frame[1])
        self.goal_in_ur_base_frame = p.multiplyTransforms(translate_inv, orient_inv, self.goal, self.start_orientation)

        if self.use_ur_only:  # observation = 3 + 6 + 6 + 3 + 3 = 21 float
            obs = np.concatenate(
                (
                    self.gripper_position,
                    self.gripper_velocity,
                    self.goal,
                    ur_joint_positions,
                    ur_joint_velocities,
                )
            ).astype(np.float32)
        else:
            # observation = 3 + 3 + 3 + 6 + 6 + 1 (wheel speed) + 1 (steering angle) + 1 (steering velocity) + 2 + 2 + 1 + 1 = 30 floats
            wheel_states = p.getJointStates(
                self.robot_id,
                [
                    self.robot_joint_info[joint][0]
                    for joint in UrRrReachEnv.RR_VELOCITY_JOINTS
                ],
            )
            steering_states = p.getJointStates(
                self.robot_id,
                [
                    i
                    for i in [
                        self.robot_joint_info[joint][0]
                        for joint in UrRrReachEnv.RR_POSITION_JOINTS
                    ]
                ],
            )
            mobile_base_state = p.getLinkState(self.robot_id, 0, computeLinkVelocity=True)
            self.robot_yaw = p.getEulerFromQuaternion(mobile_base_state[1])[2]
            yaw_velocity = mobile_base_state[7][2]
            self.robot_position = np.array(mobile_base_state[0])[:2]

            rr100_steering_angle = np.array([steering_states[0][0]])
            rr100_wheel_velocity = np.array([wheel_states[0][1]])
            rr100_steering_velocity = np.array([steering_states[0][1]])

            mobile_base_position = self.robot_position.copy()
            mobile_base_velocity = np.array(mobile_base_state[6])[:2]
            mobile_base_orientation = np.array([self.robot_yaw])
            mobile_base_angular = np.array([yaw_velocity])

            obs = np.concatenate(
                (
                    self.gripper_position,
                    self.gripper_velocity,
                    self.goal,
                    ur_joint_positions,
                    ur_joint_velocities,
                    rr100_wheel_velocity,
                    rr100_steering_angle,
                    rr100_steering_velocity,
                    mobile_base_velocity,
                    mobile_base_position,
                    mobile_base_orientation,
                    mobile_base_angular,
                )
            ).astype(np.float32)

        return obs

    def _get_info(self, error, distance):
        info = {
            "is_success": self._is_success(distance),
            "distance_error": error,
        }
        return info

    def _sample_goal(self, goal_space):
        goal = np.array(goal_space.sample())
        p.resetBasePositionAndOrientation(self.sphere, goal, self.start_orientation)
        return goal.copy()

    def _is_success(self, distance_error):
        return distance_error < self.distance_threshold

    def load_robot(self):
        self.ur_rr100_start_pos = [0, 0, 0]
        self.ur_rr100_start_orientation = self.start_orientation

        path = os.path.join(get_urdf_path(), "UR5_RR100/rr100_ur.urdf")

        self.robot_id = p.loadURDF(
            path,
            basePosition=self.ur_rr100_start_pos,
            baseOrientation=self.ur_rr100_start_orientation,
            useFixedBase=self.use_ur_only,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_IMPLICIT_CYLINDER,
        )

        self.ur_rr100_num_joints = p.getNumJoints(self.robot_id)  # 26 joints
        joint_infos = [
            p.getJointInfo(self.robot_id, id) for id in range(self.ur_rr100_num_joints)
        ]
        self.robot_joint_info = {
            joint_info[1].decode(): joint_info[0:1] + joint_info[2:]
            for joint_info in joint_infos
        }
        self.ur_joint_ids = [
            self.robot_joint_info[joint][0]
            for joint in UrRrReachEnv.UR_CONTROLLABLE_JOINTS
        ]
        self.ur_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in UrRrReachEnv.UR_CONTROLLABLE_JOINTS
        ]
        self.wheel_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in UrRrReachEnv.RR_VELOCITY_JOINTS
        ]
        self.steering_joint_forces = [
            self.robot_joint_info[joint][10 - 1]
            for joint in UrRrReachEnv.RR_POSITION_JOINTS
        ]
        self.wheel_joint_ids = [
            self.robot_joint_info[joint][0] for joint in UrRrReachEnv.RR_VELOCITY_JOINTS
        ]
        self.steering_joint_ids = [
            self.robot_joint_info[joint][0] for joint in UrRrReachEnv.RR_POSITION_JOINTS
        ]

        self.velocity_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in UrRrReachEnv.UR_CONTROLLABLE_JOINTS
                + UrRrReachEnv.RR_VELOCITY_JOINTS
            ]
        )
        self.position_limits = np.array(
            [
                self.robot_joint_info[joint][11 - 1]
                for joint in UrRrReachEnv.UR_CONTROLLABLE_JOINTS
                + UrRrReachEnv.RR_POSITION_JOINTS
            ]
        )
        
        self.ur_base_link_id = self.robot_joint_info["ur_base_joint"][0]

        for joint in UrRrReachEnv.RR_POSITION_JOINTS:
            p.changeDynamics(
                self.robot_id,
                self.robot_joint_info[joint][0],
                maxJointVelocity=self.robot_joint_info[joint][11 - 1],
            )
            p.setCollisionFilterPair(
                self.robot_id,
                self.robot_id,
                self.robot_joint_info["base_footprint_joint"][0],
                self.robot_joint_info[joint][0],
                0,
            )
        for joint in UrRrReachEnv.RR_VELOCITY_JOINTS:
            p.setCollisionFilterPair(
                self.robot_id,
                self.robot_id,
                self.robot_joint_info["base_footprint_joint"][0],
                self.robot_joint_info[joint][0],
                0,
            )

    def load_walls(self, min_x, max_x, min_y, max_y):
        path = os.path.join(get_urdf_path(), "my_cube.urdf")
        self.wall_1 = p.loadURDF(
            path,
            basePosition=[max_x + 0.25, 0, 0],
            baseOrientation=[0.0, 0.0, 0.707, 0.707],
            globalScaling=10,
            useFixedBase=True,
        )
        self.wall_2 = p.loadURDF(
            path,
            basePosition=[0, max_y + 0.25, 0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            globalScaling=10,
            useFixedBase=True,
        )
        self.wall_3 = p.loadURDF(
            path,
            basePosition=[0, min_y - 0.25, 0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            globalScaling=10,
            useFixedBase=True,
        )
        self.wall_4 = p.loadURDF(
            path,
            basePosition=[min_x - 0.25, 0, 0],
            baseOrientation=[0.0, 0.0, 0.707, 0.707],
            globalScaling=10,
            useFixedBase=True,
        )

    def set_ur_initial_joints_positions(self, init_gripper=True):
        self.ur_joint_name_to_ids = {}
        # Set intial positions
        initial_positions = {
            "ur_shoulder_pan_joint": np.pi,
            "ur_shoulder_lift_joint": -np.pi / 2,
            "ur_elbow_joint": np.pi / 2,
            "ur_wrist_1_joint": -np.pi / 2,
            "ur_wrist_2_joint": -np.pi / 2,
            "ur_wrist_3_joint": 0.0,
            "front_left_steering_joint": 0.0,
            "front_right_steering_joint": 0.0,
            "rear_left_steering_joint": 0.0,
            "rear_right_steering_joint": 0.0,
        }

        self.ur_velocity_limits = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])

        self.rr100_velocity_steering_limits = np.array(
            [np.pi, 0.57]
        )  # velocity, steering
        if self.use_ackermann:
            self.rr100_velocity_steering_limits = np.array(
                [2.0, 1.0]
            )  # velocity, steering

        indices = (
            self.robot_joint_info[joint][0] for joint in initial_positions.keys()
        )

        for id, position in zip(indices, initial_positions.values()):
            p.resetJointState(self.robot_id, id, position)

    def set_gym_spaces_ur(self):
        ur_eff_state = p.getLinkState(self.robot_id, self.arm_eef_index)
        """
        # LARGE
        low_marge = 0.1
        low_x_down = panda_eff_state[0][0]-1.5*low_marge
        low_x_up = panda_eff_state[0][0]+0.5*low_marge
        
        low_y_down = panda_eff_state[0][1]-4*low_marge
        low_y_up = panda_eff_state[0][1]+4*low_marge
        
        
        z_low_marge = 0.3
        low_z_down = panda_eff_state[0][2]-z_low_marge
        low_z_up = panda_eff_state[0][2]
        """
        """
        # MEDIUM
        low_marge = 0.1
        low_x_doselfwn = panda_eff_state[0][0]-1.5*low_marge
        low_x_up = panda_eff_state[0][0]+0.5*low_marge
        
        low_y_down = panda_eff_state[0][1]-3*low_marge
        low_y_up = panda_eff_state[0][1]+3*low_marge
        
        
        z_low_marge = 0.25
        low_z_down = panda_eff_state[0][2]-z_low_marge
        low_z_up = panda_eff_state[0][2]
        """

        self.goal_spaces = []

        # SMALL
        low_marge = 0.1
        low_x_down = ur_eff_state[0][0] - 1.0 * low_marge
        low_x_up = ur_eff_state[0][0] + 0.5 * low_marge

        low_y_down = ur_eff_state[0][1] - 2.5 * low_marge
        low_y_up = ur_eff_state[0][1] + 2.5 * low_marge

        z_low_marge = 0.25
        low_z_down = ur_eff_state[0][2] - z_low_marge
        low_z_up = ur_eff_state[0][2]

        # Smaller goal space for training, for better generalization evaluation
        self.goal_spaces.append(
            spaces.Box(
                low=np.array([low_x_down, low_y_down, low_z_down]),
                high=np.array([low_x_up, low_y_up, low_z_up]),
            )
        )

        # MEDIUM
        low_marge = 0.1
        low_x_down = ur_eff_state[0][0] - 1.5 * low_marge
        low_x_up = ur_eff_state[0][0] + 0.5 * low_marge

        low_y_down = ur_eff_state[0][1] - 3 * low_marge
        low_y_up = ur_eff_state[0][1] + 3 * low_marge

        z_low_marge = 0.25
        low_z_down = ur_eff_state[0][2] - z_low_marge
        low_z_up = ur_eff_state[0][2]

        self.goal_spaces.append(
            spaces.Box(
                low=np.array([low_x_down, low_y_down, low_z_down]),
                high=np.array([low_x_up, low_y_up, low_z_up]),
            )
        )

        # LARGE
        low_marge = 0.1
        low_x_down = ur_eff_state[0][0] - 1.5 * low_marge
        low_x_up = ur_eff_state[0][0] + 0.5 * low_marge

        low_y_down = ur_eff_state[0][1] - 4 * low_marge
        low_y_up = ur_eff_state[0][1] + 4 * low_marge

        z_low_marge = 0.3
        low_z_down = ur_eff_state[0][2] - z_low_marge
        low_z_up = ur_eff_state[0][2]

        self.goal_spaces.append(
            spaces.Box(
                low=np.array([low_x_down, low_y_down, low_z_down]),
                high=np.array([low_x_up, low_y_up, low_z_up]),
            )
        )

        low_marge = 0.1
        low_x_down = ur_eff_state[0][0] - 2 * low_marge
        low_x_up = ur_eff_state[0][0] + low_marge

        low_y_down = ur_eff_state[0][1] - 5 * low_marge
        low_y_up = ur_eff_state[0][1] + 5 * low_marge

        z_low_marge = 0.3
        low_z_down = ur_eff_state[0][2] - z_low_marge
        low_z_up = ur_eff_state[0][2]

        self.position_space = spaces.Box(
            low=np.array([low_x_down, low_y_down, low_z_down], dtype=np.float64),
            high=np.array([low_x_up, low_y_up, low_z_up], dtype=np.float64),
        )

        obs = self._get_obs()

        num_joints_ur = len(UrRrReachEnv.UR_CONTROLLABLE_JOINTS)
        # action_space = joint velocity = 6 float
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(num_joints_ur,), dtype=np.float32
        )

        # observation = 21 float -> see function _get_obs
        self.observation_space = spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            # shape=(num_joints_ur * 2 + 3 + 3 + 3,),
            shape=obs.shape,
            dtype=np.float32,
        )

    def set_gym_spaces_ur_rr100(self):
        """# Min/Max Z
        z_down = 0.64
        z_up = 1.34"""

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

        self.load_walls(x_down, x_up, y_down, y_up)

        # action_space = joint position (6 float) + mobile base velocity and steering angle (2 float) = 8 float
        self.n_actions = len(UrRrReachEnv.UR_CONTROLLABLE_JOINTS) + 2
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.n_actions,), dtype=np.float32
        )

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            shape=obs.shape,
            # shape=(num_joints_ur * 2 + num_wheels + num_joints_steering * 2 + 3 + 3 + 3 + 3 + 3,),
            dtype=np.float32,
        )

    def debug_gui(self, low, high, color=[0, 0, 1]):
        # low_array = self.pos_space.low
        # high_array = self.pos_space.high

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

    def close(self):
        p.disconnect()
