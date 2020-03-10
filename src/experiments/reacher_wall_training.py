import logging

import numpy as np

from algorithm.ppo import PPOLearner
from robot_environments.reacher_wall import ReacherWallEnv
from collections import Counter
from utils.post_processing import save_training_rewards, save_video

logger = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment
environment = ReacherWallEnv(render=False, use_naive_reward=True)

# Define action map
step_size = 3.0
action_space = [
    [0, -1],
    [0, 1],
    [1, 0],
    [-1, 0],
    [0, 0]
]
action_map = {i: step_size * np.array(action) for i, action in enumerate(action_space)}

# Initialize learner
seed = 1
learner = PPOLearner(
    environment=environment,
    state_space_dimension=3,
    action_map=action_map,
    critic_hidden_layer_units=[32, 16],
    actor_hidden_layer_units=[64, 32],
    n_iterations=75,
    n_steps_per_trajectory=200,
    n_trajectories_per_batch=20,
    seed=seed
)

# Train learner
learner.train()

# Save outputs
save_training_rewards(learner=learner, path="reacher_wall_training_rewards")
#save_video(learner=learner, path=f"reacher_wall_{seed}_video", use_argmax=False)
