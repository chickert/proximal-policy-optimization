import numpy as np
import matplotlib.pyplot as plt
import logging

from algorithm.ppo import PPOLearner
from toy_environments.goal_finder import GoalFinderEnv

logger = logging.basicConfig(level=logging.INFO)

# Constants
N_DIMENSIONS = 2
SPARSITY_PARAM = 1.0  # scaling factor for Gaussian kernel reward function
REWARD_NOISE = 0  # standard deviation of Gaussian noise added to reward function
N_TRIALS = 1
ACTOR_HIDDEN_LAYER_UNITS = [64, 32]
CRITIC_HIDDEN_LAYER_UNITS = [32, 18]
DISCRETE_STEP_SIZE = 5e-2
DISCRETE_ACTOR = False
TRAINING_REWARDS_PATH = "../../results/goal_finder/training_rewards"
TRAJECTORY_PLOTS_PATH = "../../results/goal_finder/trajectory_plot"

# Initialize environment
environment = GoalFinderEnv(
    initial_state=np.zeros(N_DIMENSIONS),
    goal_state=np.ones(N_DIMENSIONS),
    sparsity_param=SPARSITY_PARAM,
    reward_noise=REWARD_NOISE
)

# Make discrete action map (if required)
if DISCRETE_ACTOR:
    action_map = {
        key: DISCRETE_STEP_SIZE * action for key, action in enumerate(
            [np.identity(N_DIMENSIONS)[i] for i in range(N_DIMENSIONS)]
            + [-np.identity(N_DIMENSIONS)[i] for i in range(N_DIMENSIONS)]
            + [np.zeros(N_DIMENSIONS)]
        )
    }


def save_trajectory_plots(
        learner: PPOLearner,
        path: str,
        n_sample_trajectories: int = 3,
) -> None:
    fig = plt.figure(figsize=(10, 10))
    for i in range(n_sample_trajectories):
        states, _, _, _ = learner.generate_trajectory(use_argmax=False)
        plt.plot(
            [state[0] for state in states],
            [state[1] for state in states],
            color="tab:blue",
            alpha=0.5,
            label=f"sample trajectory {i+1}"
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
        s=50,
        label="initial state"
    )
    plt.scatter(
        learner.environment.goal_state[0],
        learner.environment.goal_state[1],
        marker="o",
        color="tab:green",
        s=50,
        label="goal state"
    )
    plt.legend(fontsize=14, loc="upper left")
    fig.savefig(path)


for seed in range(N_TRIALS):
    if DISCRETE_ACTOR:
        leaner = PPOLearner(
            environment=environment,
            state_space_dimension=N_DIMENSIONS,
            action_space_dimension=len(action_map),
            actor_hidden_layer_units=ACTOR_HIDDEN_LAYER_UNITS,
            critic_hidden_layer_units=CRITIC_HIDDEN_LAYER_UNITS,
            discrete_actor=True,
            action_map=action_map,
            seed=seed
        )
    else:
        learner = PPOLearner(
            environment=environment,
            state_space_dimension=N_DIMENSIONS,
            action_space_dimension=N_DIMENSIONS,
            actor_hidden_layer_units=ACTOR_HIDDEN_LAYER_UNITS,
            critic_hidden_layer_units=CRITIC_HIDDEN_LAYER_UNITS,
            seed=seed
        )
    learner.train()
    learner.save_training_rewards(path=TRAINING_REWARDS_PATH)
    if N_DIMENSIONS == 2:
        save_trajectory_plots(
            learner=learner,
            path=f"{TRAJECTORY_PLOTS_PATH}_{seed}"
        )

