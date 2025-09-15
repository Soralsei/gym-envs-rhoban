from abc import abstractmethod

import numpy as np
import gymnasium as gym


class GoalEnvMixin():
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

    def _reward_info(self) -> dict:
        """Compute info dictionary to be used by the compute_reward method"""
        return {}

    def _get_dict_obs(self) -> dict:
        """Get the current observation of the environment, including achieved and desired goals."""
        raise NotImplementedError("This method should be implemented in the subclass.")

    def step(self, action):
        """Custom Stepping function which calls compute_reward instead of our usual _reward method"""
        self._set_action(action)  # type: ignore

        self._step_simulation()  # type: ignore

        obs = self._get_dict_obs()

        info = self._reward_info()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        info.update(self._get_info())  # type: ignore

        terminated = self._is_terminal()  # type: ignore
        truncated = self._should_truncate()  # type: ignore

        return obs, reward, terminated, truncated, info
    
    def get_obs(self):
        """Get the current observation of the environment."""
        return self._get_dict_obs()
        
