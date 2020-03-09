import logging

import numpy as np

from algorithm.ppo import PPOLearner
from robot_environments.reacher_wall import ReacherWallEnv
from utils.post_processing import save_training_rewards, save_video

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = ReacherWallEnv(render=False, use_naive_reward=False)

# Define action map
step_size = 3.0
action_space = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, -1],
    [0, 1],
]
action_map = {i: step_size * np.array(action) for i, action in enumerate(action_space)}

# Run training over multiple random seeds
for seed in [123]:

    # Initialize learner
    learner = PPOLearner(
        environment=environment,
        state_space_dimension=3,
        action_map=action_map,
        critic_hidden_layer_units=[32, 16],
        actor_hidden_layer_units=[128, 64],
        n_iterations=20,
        n_steps_per_trajectory=200,
        n_trajectories_per_batch=10,
        learning_rate=5e-4,
        n_epochs=5,
        seed=seed
    )

    # Train learner
    learner.train()

    # Save outputs
    #save_training_rewards(learner=learner, path="reacher_training_rewards")
    for i in range(2):
        save_video(learner=learner, path=f"reacher_wall_{seed}_random_video_{i}")
