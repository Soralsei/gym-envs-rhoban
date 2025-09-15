from typing import Any, Iterable
from abc import abstractmethod, ABC

import gymnasium as gym

import numpy as np


class BaseEnv(gym.Env, ABC):

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_actions: int,
        render_mode="human",
        agent_action_frequency: int = 40,
    ):
        self.n_actions = n_actions
        self.render_mode = render_mode

        self.robot_action_frequency = agent_action_frequency  # Hz
        self.action_dt = 1 / self.robot_action_frequency  # s

        self.total_episodes = 0

        # self._init_simulation()

    # basic methods
    # -------------------------
    def step(self, action):
        print("Calling step in BaseEnv")
        self._set_action(action)

        self._step_simulation()

        obs = self._get_obs()

        reward = self._reward()
        info = self._get_info()

        terminated = self._is_terminal()
        truncated = self._should_truncate()

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        print("Calling reset in BaseEnv")
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._seed(seed)

        # Call child reset method
        self._reset(options=options or {})

        obs = self.get_obs()
        info = self._get_info()

        self.total_episodes += 1

        return obs, info

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            return self._render_human()

        if self.render_mode == "rgb_array":
            # (width, height, rgbPixels, depthPixels, segmentationMaskBuffer)
            return self._render_rgb_array()
        
    def set_gym_spaces(self) -> None:
        self._set_goal_space()
        self._set_action_space()
        self._set_observation_space()

    def close(self) -> None:
        self._close()

    @abstractmethod
    def _init_simulation(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _seed(self, seed) -> None:
        raise NotImplementedError

    @abstractmethod
    def _reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _step_simulation(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _reward(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def _set_action(self, action: Iterable) -> None:
        raise NotImplementedError

    def get_obs(self):
        """Get the current observation of the environment."""
        return self._get_obs()

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _get_info(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _is_terminal(self) -> bool:
        raise NotImplementedError

    def _should_truncate(self) -> bool:
        return False

    @abstractmethod
    def _render_rgb_array(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _render_human(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_goal_space(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_action_space(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_observation_space(self) -> None:
        raise NotImplementedError
