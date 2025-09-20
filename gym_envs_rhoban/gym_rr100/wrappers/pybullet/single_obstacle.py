import os
from copy import deepcopy
import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from gymnasium import Wrapper

from gym_envs_rhoban import get_urdf_path


class SingleObstacleWrapper(Wrapper):
    """
    Wrapper that adds a single spherical obstacle to the environment.
    The obstacle is placed randomly around the goal position at each reset.
    The obstacle position is provided in the observation.
    A collision with the obstacle results in a penalty in the reward.

    Note: this wrapper assumes that the returned reward will not be overwritten (completely replaced) later by other wrappers.
    """

    def __init__(
        self,
        env: gym.Env,
        resample_obstacle: bool = True,
        initial_obstacle_position=None,
    ):
        super().__init__(env)
        self._observation_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self.env.observation_space.shape[0] + 2,),  # type: ignore # Add 2 for the obstacle position
            dtype=np.float32,
        )

        obstacle_path = os.path.join(get_urdf_path(), "obstacle_sphere.urdf")
        self.obstacle_sphere = p.loadURDF(
            obstacle_path,
            basePosition=[0.0, 0.0, -10],
            globalScaling=1,
            useFixedBase=True,
        )

        self.safety_distance = self.env.unwrapped.wheel_base + self.env.unwrapped.wheel_radius + 0.4 # type: ignore
        self.obstacle = initial_obstacle_position or [0.0, 0.0]
        # if initial_obstacle_position is None:
        #     self.obstacle = self._sample_obstacle_position()
        self.place_obstacle(self.obstacle)  # type: ignore
        self.resample_obstacle = resample_obstacle

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.resample_obstacle:
            self.obstacle = self._sample_obstacle_position()
            self.place_obstacle(self.obstacle)  # type: ignore
        else:
            # Ensure obstacle is not too close to goal
            while np.linalg.norm(self.env.unwrapped.goal - self.obstacle) < self.safety_distance:  # type: ignore
                print("Warning: obstacle too close to goal, resampling")
                options = {}
                if kwargs is not None:
                    options = kwargs.get("options") or {}
                print("Options:", options)
                self.env.unwrapped._reset(options=options)  # type: ignore

        obs = np.concatenate([obs, self.obstacle])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.concatenate([obs, self.obstacle])
        return obs, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        distance_to_obstacle = p.getClosestPoints(
            self.env.unwrapped.robot_id, self.obstacle_sphere, 10
        )
        if distance_to_obstacle:
            self.distance_to_obstacle = distance_to_obstacle = distance_to_obstacle[0][
                8
            ]  # Closest distance of first point
        else:
            self.distance_to_obstacle = 10.0  # No obstacle within 10 units

        if self.distance_to_obstacle < 0.05:
            reward -= 5.0  # Collision penalty

        return reward

    def _sample_obstacle_position(self):
        obstacle = np.zeros(2)
        safety_distance = (
            self.env.unwrapped.wheel_base + self.env.unwrapped.wheel_radius + 0.4  # type: ignore
        )
        obstacle_max_dist = 1.5  # meters

        sample_space = spaces.Box(
            low=np.array([0.4, 0.0]),
            high=np.array([obstacle_max_dist, 2 * np.pi]),
        )  # Polar coordinates

        norm = 0.0
        obstacle = [float("nan"), float("nan")]

        while norm < safety_distance:
            r, theta = sample_space.sample()
            obstacle = np.array(
                [r * np.cos(theta), r * np.sin(theta)]
            )  # Convert back to cartesian coordinates
            obstacle = obstacle + self.env.unwrapped.goal  # type: ignore # Translate obstacle to be around the goal
            norm = np.linalg.norm(obstacle - self.env.unwrapped.pos_of_interest)  # type: ignore # Check if obstacle is far enough from robot

        return obstacle

    def place_obstacle(self, obstacle_position):
        p.resetBasePositionAndOrientation(
            self.obstacle_sphere,
            [obstacle_position[0], obstacle_position[1], -15],
            [0, 0, 0, 1],
        )


class SingleObstacleDictWrapper(Wrapper):
    """
    Wrapper that adds a single spherical obstacle to the environment.
    The obstacle is placed randomly around the goal position at each reset.
    The obstacle position is provided in the observation.
    A collision with the obstacle results in a penalty in the reward.

    Note: this wrapper assumes that the returned reward will not be overwritten (completely replaced) later by other wrappers.
    """

    def __init__(
        self,
        env: gym.Env,
        resample_obstacle: bool = True,
        initial_obstacle_position=None,
    ):
        super().__init__(env)
        if not isinstance(self.env.observation_space, spaces.Dict):
            raise ValueError(
                "SingleObstacleWrapper only supports Dict observation spaces"
            )

        self._observation_space = deepcopy(self.env.observation_space)
        self._observation_space.spaces["observation"] = spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            shape=(self._observation_space.spaces["observation"].shape[0] + 2,),
            dtype=np.float32,
        )

        obstacle_path = os.path.join(get_urdf_path(), "obstacle_sphere.urdf")
        self.obstacle_sphere = p.loadURDF(
            obstacle_path,
            basePosition=[0.0, 0.0, -10],
            globalScaling=1,
            useFixedBase=True,
        )

        self.obstacle = initial_obstacle_position or [0.0, 0.0]
        # if initial_obstacle_position is None:
        #     self.obstacle = self._sample_obstacle_position()
        self.place_obstacle(self.obstacle)  # type: ignore
        print("Obstacle ", self.obstacle)
        self.resample_obstacle = resample_obstacle

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.resample_obstacle:
            self.obstacle = self._sample_obstacle_position()
            self.place_obstacle(self.obstacle)  # type: ignore
        else:
            # Ensure obstacle is not too close to goal
            safety_distance = self.env.unwrapped.wheel_base + self.env.unwrapped.wheel_radius + 0.4  # type: ignore
            while np.linalg.norm(self.env.unwrapped.goal - self.obstacle) < safety_distance:  # type: ignore
                print("Warning: obstacle too close to goal, resampling")
                options = {}
                if kwargs is not None:
                    options = kwargs.get("options") or {}
                print("Options:", options)
                self.env.unwrapped._reset(options=options)  # type: ignore
        print("Obstacle ", self.obstacle)
        obs["observation"] = np.concatenate([obs["observation"], self.obstacle])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["observation"] = np.concatenate([obs["observation"], self.obstacle])
        return obs, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        distance_to_obstacle = p.getClosestPoints(
            self.env.unwrapped.robot_id, self.obstacle_sphere, 10
        )
        if distance_to_obstacle:
            self.distance_to_obstacle = distance_to_obstacle = distance_to_obstacle[0][
                8
            ]  # Closest distance of first point
        else:
            self.distance_to_obstacle = 10.0  # No obstacle within 10 units

        if self.distance_to_obstacle < 0.05:
            reward -= 5.0  # Collision penalty

        return reward

    def _sample_obstacle_position(self):
        obstacle = np.zeros(2)
        safety_distance = (
            self.env.unwrapped.wheel_base + self.env.unwrapped.wheel_radius + 0.4  # type: ignore
        )
        obstacle_max_dist = 1.5  # meters

        sample_space = spaces.Box(
            low=np.array([0.4, 0.0]),
            high=np.array([obstacle_max_dist, 2 * np.pi]),
        )  # Polar coordinates

        norm = 0.0
        obstacle = [float("nan"), float("nan")]

        while norm < safety_distance:
            r, theta = sample_space.sample()
            obstacle = np.array(
                [r * np.cos(theta), r * np.sin(theta)]
            )  # Convert back to cartesian coordinates
            obstacle = obstacle + self.env.unwrapped.goal  # type: ignore # Translate obstacle to be around the goal
            norm = np.linalg.norm(obstacle - self.env.unwrapped.pos_of_interest)  # type: ignore # Check if obstacle is far enough from robot

        return obstacle

    def place_obstacle(self, obstacle_position):
        p.resetBasePositionAndOrientation(
            self.obstacle_sphere,
            [obstacle_position[0], obstacle_position[1], 0.5 + 1e-3],
            # [obstacle_position[0], obstacle_position[1], -15],
            [0, 0, 0, 1],
        )
