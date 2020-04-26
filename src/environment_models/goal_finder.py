import numpy as np
import matplotlib.pyplot as plt
from environment_models.base import BaseEnv
from algorithms.ppo import PPOLearner

from typing import Callable


class GoalFinderEnv(BaseEnv):

    def __init__(
            self,
            n_dimensions: int = 2,
            sparsity_param: float = 1.0,
            reward_noise: float = 0.0,
            noise_sample: Callable = np.random.randn,
            rescale_sparsity_param: bool = True,
            seed: int = 0
    ):
        initial_state = np.zeros(n_dimensions)
        goal_state = np.ones(n_dimensions)

        if rescale_sparsity_param:
            sparsity_param = sparsity_param / np.sqrt(n_dimensions)

        def transition_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> np.ndarray:
            return state + action

        def reward_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> float:
            distance_to_goal = np.linalg.norm(state - goal_state, 2)
            return np.exp(-sparsity_param * distance_to_goal ** 2) + reward_noise*noise_sample()

        BaseEnv.__init__(
            self,
            initial_state=initial_state,
            transition_function=transition_function,
            reward_function=reward_function
        )

        np.random.seed(seed)
        self.goal_state = goal_state


def save_2d_trajectory_plot(
        learner: PPOLearner,
        path: str,
) -> None:
    assert len(learner.environment.state) == 2, ValueError
    fig = plt.figure(figsize=(10, 10))
    states, _, _, _ = learner.generate_trajectory(use_argmax=False)
    plt.plot(
        [state[0] for state in states],
        [state[1] for state in states],
        color="tab:blue",
        alpha=0.5,
        label=f"sample trajectory"
    )
    states, _, _, _ = learner.generate_trajectory(use_argmax=True)
    plt.plot(
        [state[0] for state in states],
        [state[1] for state in states],
        color="tab:orange",
        alpha=0.8,
        label="argmax trajectory"
    )
    plt.scatter(
        learner.environment.initial_state[0],
        learner.environment.initial_state[1],
        marker="o",
        color="k",
        label="initial state"
    )
    plt.scatter(
        learner.environment.goal_state[0],
        learner.environment.goal_state[1],
        marker="o",
        color="tab:green",
        label="goal state"
    )
    plt.legend(fontsize=14, loc="upper left")
    fig.savefig(path)
    plt.close()
