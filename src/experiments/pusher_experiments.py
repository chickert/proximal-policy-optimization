import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset

from algorithms.ppo import PPOLearner
from environment_models.pusher import PusherEnv

logger = logging.basicConfig(level=logging.DEBUG)

# Set experiment parameters
EXPERT_DATA_PATH = "../../data/expert.npz"
RESULTS_FOLDER = "../../data/results/pusher/"
N_TRIALS = 5
PPO_PARAMS = {
    "action_space_dimension": 2,
    "actor_hidden_layer_units": [128, 64],
    "critic_hidden_layer_units": [64, 32],
    "n_steps_per_trajectory": 16,
    "n_trajectories_per_batch": 64,
    "n_iterations": 50,
    "learning_rate": 3e-4,
    "entropy_coefficient": 0,
    "actor_std": 5e-3,
    "n_epochs": 3
}


if __name__ == "__main__":

    # Load data
    expert_data = np.load(EXPERT_DATA_PATH)
    expert_data = TensorDataset(torch.tensor(expert_data["obs"]), torch.tensor(expert_data["action"]))
    states, actions = map(torch.stack, zip(*expert_data))
    print(states.shape, actions.shape)



    # # Intialze environment
    # environment = PusherEnv()
    #
    # # Learn
    # learner = PPOLearner(
    #     environment=environment,
    #     **PPO_PARAMS
    # )
    # learner.train(expert_data=expert_data)


