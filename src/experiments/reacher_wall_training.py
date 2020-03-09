import logging

import numpy as np

from algorithm.ppo import PPOLearner
from robot_environments.reacher_wall import ReacherWallEnv
from utils.post_processing import save_training_rewards

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = ReacherWallEnv(render=False)

# Define action map
step_size = 15.0
action_space = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
]
action_map = {i: step_size * np.array(action) for i, action in enumerate(action_space)}

#
good_seeds = [1] #[1, 8, 10, 12]
for seed in good_seeds:
    # Initialize learner
    learner = PPOLearner(
        environment=environment,
        state_space_dimension=3,
        action_map=action_map,
        critic_hidden_layer_units=[32, 16],
        actor_hidden_layer_units=[64, 32],
        n_iterations=50,
        init_reversion_threshold=1.0,
        min_reversion_threshold=0.5,
        seed=seed
    )

    # Train learner
    learner.train()

    # L
    learner.environment = ReacherWallEnv(render=True)
    _ = learner.generate_argmax_trajectory()

    # Save outputs
    save_training_rewards(learner=learner, path=f"reacher_training_rewards")
    save_videos(learner=learner, path=f"reacher_{seed}_video")
