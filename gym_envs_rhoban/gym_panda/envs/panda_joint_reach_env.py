from enum import IntEnum
import os
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data as pd

from gym_envs import get_urdf_path


class GoalSpaceSize(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


class JointPandaReachEnv(gym.Env):

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}
    
    CONTROLLABLE_JOINTS : list[str] = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6"
    ]

    def __init__(
        self,
        render_mode="human",
        goal_space_size: GoalSpaceSize = GoalSpaceSize.SMALL,
        distance_threshold: float = 0.05,
    ):
        # switch tool to convert a numeric value to string value
        self.switcher_type_name = {
            p.JOINT_REVOLUTE: "JOINT_REVOLUTE",
            p.JOINT_PRISMATIC: "JOINT_PRISMATIC",
            p.JOINT_SPHERICAL: "JOINT SPHERICAL",
            p.JOINT_PLANAR: "JOINT PLANAR",
            p.JOINT_FIXED: "JOINT FIXED",
        }

        self.total_episodes = 0
        self.goal_space_size = goal_space_size
        
        # bullet parameters
        self.time_step = 1.0 / 240
        self.n_substeps = 20
        self.dt = self.time_step * self.n_substeps

        # robot parameters
        self.distance_threshold = distance_threshold
        self.arm_eef_index = 11
        self.max_vel = 1
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # connect bullet
        simulation_type = p.DIRECT if render_mode != "human" else p.GUI
        p.connect(
            simulation_type
        )  # or p.GUI (for test) or p.DIRECT (for train) for non-graphical version
        self.if_render = False
        self.render_mode = render_mode

        self._init_simulation()

        self.debug_gui(self.pos_space.low, self.pos_space.high, [0, 0, 1])
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

    def printAllInfo(self):
        print("=================================")
        print("All Robot joints info")
        num_joints = p.getNumJoints(self.panda_id)
        print("=> num of joints = {0}".format(num_joints))
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.panda_id, i)
            # print(joint_info)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]
            child_link_name = joint_info[12].decode("UTF-8")
            link_pos_in_parent_frame = p.getLinkState(self.panda_id, i)[0]
            link_orient_in_parent_frame = p.getLinkState(self.panda_id, i)[1]
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
                    link_orient_in_parent_frame,
                )
            )
        print("=================================")

    def _init_simulation(self):
        p.resetSimulation()

        # bullet setup
        # self.seed()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(numSubSteps=self.n_substeps)

        p.setGravity(0, 0, -9.81)

        # load table
        self.table_pos = [0, 0, 0]
        self.table_height = 0.625
        self.table = p.loadURDF("table/table.urdf", self.table_pos, useFixedBase=True)

        # load panda
        self.load_panda()
        p.stepSimulation()

        # set panda joints to initial positions
        self.set_panda_initial_joints_positions()
        p.stepSimulation()

        # set gym spaces
        self.set_gym_spaces()
        p.stepSimulation()

        # load goal
        fullpath = os.path.join(get_urdf_path(), "my_sphere.urdf")
        self.sphere = p.loadURDF(fullpath, useFixedBase=True)
        p.stepSimulation()

    # basic methods
    # -------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        
        p.stepSimulation()

        obs = self._get_obs()

        error = np.linalg.norm(obs[12:15] - self.goal)
        terminated = self._is_success(error)
        reward = -error
        truncated = False
        
        info = self._get_info(error, terminated)

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
        self.set_panda_initial_joints_positions()
        p.stepSimulation()

        goal_space = self.goal_spaces[self.goal_space_size]
        self.goal = self._sample_goal(goal_space)

        obs = self._get_obs()
        error = np.linalg.norm(obs[12:15] - self.goal)
        info = self._get_info(error, False)
        
        self.total_episodes += 1

        return obs, info

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            return None

        if self.render_mode == "rgb_array":
            return p.getCameraImage(640, 360)[
                2
            ]  # (width, height, rgbPixels, depthPixels, segmentationMaskBuffer)

    # RobotEnv method
    # -------------------------
    def _set_action(self, action):
        #action is the joint velocity
        assert action.shape == (6,), "Action shape error"
        
        clipped_action = np.clip(action * self.velocity_limits, -self.velocity_limits, self.velocity_limits)
        indices = (self.panda_joint_info[joint][0] for joint in JointPandaReachEnv.CONTROLLABLE_JOINTS)
        
        for id, control in zip(indices, clipped_action):
            p.setJointMotorControl2(
                self.panda_id, id, p.VELOCITY_CONTROL, targetVelocity=control, force=5 * 240.0
            )

    def _get_obs(self):
        joint_states = p.getJointStates(self.panda_id, range(6))
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        ee_state = p.getLinkState(
            self.panda_id, self.arm_eef_index, computeLinkVelocity=1
        )
        grip_pos = np.array(ee_state[0])
        grip_velp = np.array(ee_state[6])
        # observation = 6 + 6 + 3 + 3 + 3 = 21 float
        obs = np.concatenate((joint_positions, joint_velocities, grip_pos, grip_velp, self.goal)).astype(np.float32)

        return obs

    def _get_info(self, error, success):
        info = {
            "is_success": success,
            "distance_error": error,
        }
        return info

    def _sample_goal(self, goal_space):
        goal = np.array(goal_space.sample())
        p.resetBasePositionAndOrientation(self.sphere, goal, self.startOrientation)
        return goal.copy()

    def _is_success(self, distance_error):
        return distance_error < self.distance_threshold

    def load_panda(self):
        # load panda
        self.panda_start_pos = [0.7, 0, self.table_pos[2] + self.table_height]
        self.panda_start_orientation = self.startOrientation

        path = os.path.join(get_urdf_path(), "franka_panda/panda.urdf")
        self.panda_id = p.loadURDF(
            path,
            basePosition=self.panda_start_pos,
            baseOrientation=self.panda_start_orientation,
            useFixedBase=True,
        )

        self.panda_num_joints = p.getNumJoints(self.panda_id)  # 12 joints
        panda_joint_info = [p.getJointInfo(self.panda_id, id) for id in range(self.panda_num_joints)]
        self.panda_joint_info = {joint_info[1].decode() : joint_info[0:1] + joint_info[2:] for joint_info in panda_joint_info}

        print("panda num joints = {}".format(self.panda_num_joints))

        self.velocity_limits = np.array([self.panda_joint_info[joint][11 - 1] for joint in JointPandaReachEnv.CONTROLLABLE_JOINTS])

    def set_panda_initial_joints_positions(self, init_gripper=True):
        self.panda_joint_name_to_ids = {}

        # Set intial positions
        panda_initial_positions = {
            "panda_joint1": 0.0,
            "panda_joint2": 0.287,
            "panda_joint3": 0.0,
            "panda_joint4": -2.392,
            "panda_joint5": 0.0,
            "panda_joint6": 2.646,
            "panda_joint7": -0.799,
            "panda_finger_joint1": 0.08,
            "panda_finger_joint2": 0.08,
        }
        
        indices = (self.panda_joint_info[joint][0] for joint in panda_initial_positions.keys())

        for id, position in zip(indices, panda_initial_positions.values()):
            p.resetJointState(self.panda_id, id, position)

    def set_gym_spaces(self):
        panda_eff_state = p.getLinkState(self.panda_id, self.arm_eef_index)
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
        
        # MEDIUM
        low_marge = 0.1
        low_x_down = panda_eff_state[0][0]-1.5*low_marge
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
        low_x_down = panda_eff_state[0][0] - 1.0 * low_marge
        low_x_up = panda_eff_state[0][0] + 0.5 * low_marge

        low_y_down = panda_eff_state[0][1] - 2.5 * low_marge
        low_y_up = panda_eff_state[0][1] + 2.5 * low_marge

        z_low_marge = 0.25
        low_z_down = panda_eff_state[0][2] - z_low_marge
        low_z_up = panda_eff_state[0][2]

        # Smaller goal space for training, for better generalization evaluation
        self.goal_spaces.append(
            spaces.Box(
                low=np.array([low_x_down, low_y_down, low_z_down]),
                high=np.array([low_x_up, low_y_up, low_z_up]),
            )
        )

        # MEDIUM
        low_marge = 0.1
        low_x_down = panda_eff_state[0][0] - 1.5 * low_marge
        low_x_up = panda_eff_state[0][0] + 0.5 * low_marge

        low_y_down = panda_eff_state[0][1] - 3 * low_marge
        low_y_up = panda_eff_state[0][1] + 3 * low_marge

        z_low_marge = 0.25
        low_z_down = panda_eff_state[0][2] - z_low_marge
        low_z_up = panda_eff_state[0][2]

        self.goal_spaces.append(
            spaces.Box(
                low=np.array([low_x_down, low_y_down, low_z_down]),
                high=np.array([low_x_up, low_y_up, low_z_up]),
            )
        )

        # LARGE
        low_marge = 0.1
        low_x_down = panda_eff_state[0][0] - 2 * low_marge
        low_x_up = panda_eff_state[0][0] + low_marge

        low_y_down = panda_eff_state[0][1] - 5 * low_marge
        low_y_up = panda_eff_state[0][1] + 5 * low_marge

        z_low_marge = 0.3
        low_z_down = panda_eff_state[0][2] - z_low_marge
        low_z_up = panda_eff_state[0][2]

        self.goal_spaces.append(
            spaces.Box(
                low=np.array([low_x_down, low_y_down, low_z_down]),
                high=np.array([low_x_up, low_y_up, low_z_up]),
            )
        )

        low_marge = 0.1
        low_x_down = panda_eff_state[0][0] - 2 * low_marge
        low_x_up = panda_eff_state[0][0] + low_marge

        low_y_down = panda_eff_state[0][1] - 5 * low_marge
        low_y_up = panda_eff_state[0][1] + 5 * low_marge

        z_low_marge = 0.3
        low_z_down = panda_eff_state[0][2] - z_low_marge
        low_z_up = panda_eff_state[0][2]

        self.pos_space = spaces.Box(
            low=np.array([low_x_down, low_y_down, low_z_down], dtype=np.float64),
            high=np.array([low_x_up, low_y_up, low_z_up], dtype=np.float64),
        )

        # action_space = cartesian world velocity (vx, vy, vz)  = 3 float
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        # observation = 9 float -> see function _get_obs
        self.observation_space = spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            shape=(21,),
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
