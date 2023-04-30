from typing import Generic, TypeVar, Any, SupportsFloat

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class RewardsEnv(Generic[ObsType, ActType]):
    """
    rewards_envs generic class for implementing and integrating the rewards.ai agents  environment. An environment can be partially or fully observed by
    an agent. A single RewardsEnv supports for a single agent environment. Although support for multi-agents interacting in a single environment will
    there in future releases.

    Note:
        RewardsEnv follows the open source standards and patterns similar to [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium/)

    The main API methods that are implemented in this class are:

    - :meth: `reset` - Resets the environment with the very initil state. This is called after completion of each episode or before calling method: step.
    This returns two things. The current environment state/observation and the environment info.

    - :meth: `step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions,
      if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info etc.

    - :meth:`render` - Renders the environments to help visualise what the agent see, examples modes are "human", "rgb_array", "ansi" for text.
    - :meth:`close` - Closes the environment, important when external software is used, i.e. pygame for rendering, databases

    Environments have additional attributes for users to understand the implementation

    - :attr: `current_observation` - Returns the current observation of that environment
    - :attr:`action_space` - (TBD) The Space object corresponding to valid actions, all valid actions should be contained within the space.
    - :attr:`observation_space` - (TBD) The Space object corresponding to valid observations, all valid observations should be contained within the space.
    - :attr:`reward_range` - (TBD) A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode.
      The default reward range is set to :math:`(-\infty,+\infty)`.
      ``super().reset(seed=seed)`` and when assessing ``self.np_random``.
    """

    def reset(self) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to the initial state. Returning the initial observation and the encvironment info
        Args:
            None

        Returns:
            observation (ObsType) : Observation of the initial state. This will be an element of :attr:`observation_space` which is
            typically a numpy.ndarray, which can be of any form, image, or a different type of observation.

            info (dictionary) : It provides the information of the given environment
        """

        raise NotImplementedError

    def close(self) -> None:
        """
        After the user has finished using the environment, it closes the code necesary to "clean up"
        the environment.
        """
        pass

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, dict[str, Any]]:
        """
        Runs one timestep of the environment's dynamics using the agent's action. When the end
        of the episode is reached, it is necessary to call :meth:`reset` to reset the environment's
        state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.

            reward (SupportsFloat): The reward as a result of taking the action.

            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.

            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
        """

        raise NotImplementedError

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment
        rewards.ai only supports two types of render method in the current version.

        - rgb_array : This returns a numpy array of the image of the given current state
        - human : Current version of rewards.ai returns the pugame window
        """
        raise NotImplementedError

    @property
    def current_observation(self) -> ObsType:
        """
        Returns the current observation of a particular timestamp
        """
        raise NotImplementedError

    @property
    def current_action(self) -> ActType:
        """
        Returns the current action that has been taken by the agent.
        """
        raise NotImplementedError
