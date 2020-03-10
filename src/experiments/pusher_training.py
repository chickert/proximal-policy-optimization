import logging

import numpy as np

from algorithm.ppo import PPOLearner
from robot_environments.pusher import PusherEnv
from utils.post_processing import save_training_rewards, save_video

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = PusherEnv(render=False)

# Define action map
step_size = 2.0
action_space = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
]
action_map = {i: step_size * np.array(action) for i, action in enumerate(action_space)}

# Initialize learner
seed = 4
learner = PPOLearner(
    environment=environment,
    state_space_dimension=6,
    action_map=action_map,
    critic_hidden_layer_units=[32, 16],
    actor_hidden_layer_units=[128, 64],
    n_iterations=50,
    n_steps_per_trajectory=100,
    n_trajectories_per_batch=20,
    seed=seed
)

# Train learner
learner.train()

# Save outputs
save_training_rewards(learner=learner, path="pusher_training_rewards")
save_video(learner=learner, path=f"pusher_{seed}_video_1", use_argmax=False)
save_video(learner=learner, path=f"pusher_{seed}_video_2", use_argmax=False)
save_video(learner=learner, path=f"pusher_{seed}_video_3", use_argmax=False)