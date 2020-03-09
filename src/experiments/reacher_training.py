import logging

import numpy as np
import torch

from algorithm.ppo import PPOLearner
from robot_environments.reacher import ReacherEnv
from utils.post_processing import save_training_rewards, save_video

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = ReacherEnv(render=False)

# Define action map function
step_size = 20.0
action_space = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, -1],
    [0, 1],
]
action_map = {i: step_size * np.array(action) for i, action in enumerate(action_space)}

# Run training over multiple random seeds
for seed in range(1, 2):

    # Initialize learner
    learner = PPOLearner(
        environment=environment,
        state_space_dimension=3,
        action_space_dimension=2,
        critic_hidden_layer_units=[32, 16],
        actor_hidden_layer_units=[64, 32],
        n_iterations=75,
        seed=seed
    )

    # Train learner
    learner.train()

    # Save outputs
    save_training_rewards(learner=learner, path="reacher_training_rewards")
    save_video(learner=learner, path=f"reacher_{seed}_argmax_video", use_argmax=True)
    for i in range(5):
        save_video(learner=learner, path=f"reacher_{seed}_random_video_{i}", use_argmax=False)
