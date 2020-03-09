import logging

import numpy as np

from algorithm.ppo import PPOLearner
from robot_environments.reacher import ReacherEnv
from utils.post_processing import save_training_rewards, save_video

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = ReacherEnv(render=False)

# Define action map
step_size = 5.0
action_space = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
]
action_map = {i: step_size * np.array(action) for i, action in enumerate(action_space)}

# Run training over multiple random seeds
seeds = [1] # [1, 8, 10, 12, 0]
for seed in [1]:

    # Initialize learner
    learner = PPOLearner(
        environment=environment,
        action_map=action_map,
        critic_hidden_layer_units=[32, 16],
        actor_hidden_layer_units=[64, 32],
        n_iterations=100,
        seed=seed
    )

    # Train learner
    learner.train()

    # Save outputs
    #save_training_rewards(learner=learner, path="reacher_training_rewards")
    save_video(learner=learner, path=f"reacher_{seed}_argmax_video", use_argmax=True)
    save_video(learner=learner, path=f"reacher_{seed}_random_video", use_argmax=False)
