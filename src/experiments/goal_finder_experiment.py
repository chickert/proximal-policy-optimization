import numpy as np
import matplotlib.pyplot as plt
import logging

from typing import Optional
from algorithm.ppo import PPOLearner
from toy_environments.goal_finder import GoalFinderEnv

logger = logging.basicConfig(level=logging.INFO)

# Constants
REWARD_NOISE_GRID = np.linspace(0, 0.8, 9)  # standard deviation of Gaussian noise added to reward function
ACTOR_HIDDEN_LAYER_UNITS = [64, 32]
CRITIC_HIDDEN_LAYER_UNITS = [32, 18]
TRAINING_REWARDS_PATH = "../../results/goal_finder/training_rewards"
TRAJECTORY_PLOTS_PATH = "../../results/goal_finder/trajectory_plot"


def save_trajectory_plots(
        learner: PPOLearner,
        path: str,
        n_sample_trajectories: int = 3,
) -> None:
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


def run_experiment(
        training_rewards_path: str,
        trajectory_plots_path: Optional[str] = None,
        n_dimensions: int = 2,
        sparsity_param: float = 1.0,
        reward_noise: float = 0.0,
        n_trials: int = 5,
        clipping_param: float = 0.2,
        discrete_actor: bool = False,
        discrete_step_size: float = 5e-2
) -> None:

    # Initialize environment
    environment = GoalFinderEnv(
        initial_state=np.zeros(n_dimensions),
        goal_state=np.ones(n_dimensions),
        sparsity_param=sparsity_param,
        reward_noise=reward_noise,
    )

    for seed in range(n_trials):

        # Initialize learner
        if discrete_actor:
            action_map = {
                key: discrete_step_size * action for key, action in enumerate(
                    [np.identity(n_dimensions)[i] for i in range(n_dimensions)]
                    + [-np.identity(n_dimensions)[i] for i in range(n_dimensions)]
                    + [np.zeros(n_dimensions)]
                )
            }
            leaner = PPOLearner(
                environment=environment,
                state_space_dimension=n_dimensions,
                action_space_dimension=len(action_map),
                actor_hidden_layer_units=ACTOR_HIDDEN_LAYER_UNITS,
                critic_hidden_layer_units=CRITIC_HIDDEN_LAYER_UNITS,
                discrete_actor=True,
                action_map=action_map,
                clipping_param=clipping_param,
                seed=seed
            )
        else:
            learner = PPOLearner(
                environment=environment,
                state_space_dimension=n_dimensions,
                action_space_dimension=n_dimensions,
                actor_hidden_layer_units=ACTOR_HIDDEN_LAYER_UNITS,
                critic_hidden_layer_units=CRITIC_HIDDEN_LAYER_UNITS,
                clipping_param=clipping_param,
                seed=seed
            )

        # Train learner
        learner.train()

        # Save learner outputs
        learner.save_training_rewards(path=training_rewards_path)
        if (n_dimensions == 2) and trajectory_plots_path:
            save_trajectory_plots(
                learner=learner,
                path=f"{trajectory_plots_path}_seed_{seed}"
            )


if __name__ == "__main__":

    for reward_noise_ in REWARD_NOISE_GRID:

        run_experiment(
            training_rewards_path=f"{TRAINING_REWARDS_PATH}_noise_{int(10 * reward_noise_)}",
            trajectory_plots_path=f"{TRAJECTORY_PLOTS_PATH}_noise_{int(10 * reward_noise_)}",
            reward_noise=reward_noise_
        )
