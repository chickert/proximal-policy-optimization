import logging

import numpy as np
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

# Initialize learner
seed = 0
learner = PPOLearner(
    environment=environment,
    state_space_dimension=3,
    action_map=action_map,
    critic_hidden_layer_units=[16, 8],
    actor_hidden_layer_units=[64, 32],
    n_steps_per_trajectory=200,
    n_trajectories_per_batch=20,
    n_epochs=4,
    n_iterations=75,
    seed=seed
)

# Train learner
learner.train()

# Save outputs
save_training_rewards(learner=learner, path="reacher_training_rewards")
save_video(learner=learner, path=f"reacher_{seed}_video")

