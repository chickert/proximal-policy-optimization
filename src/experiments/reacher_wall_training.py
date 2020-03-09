import logging

import numpy as np

from algorithm.ppo import PPOLearner
from robot_environments.reacher_wall import ReacherWallEnv
from utils.post_processing import save_training_rewards, save_video

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = ReacherWallEnv(render=True, use_naive_reward=False)

# Define action map
step_size = 2.0
action_space = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, -1],
    [0, 1],
]
def action_map(policy_probabilities):
    action_map_dict = {i: step_size * np.array(action) for i, action in enumerate(action_space)}

# Run training over multiple random seeds
for seed in [9000]:

    # Initialize learner
    learner = PPOLearner(
        environment=environment,
        state_space_dimension=3,
        action_map=action_map,
        critic_hidden_layer_units=[32, 16],
        actor_hidden_layer_units=[64, 32],
        n_iterations=20,
        n_trajectories_per_batch=20,
        learning_rate=3e-4,
        n_epochs=3,
        seed=seed
    )

    # Train learner
    learner.train()

    # Save outputs
    #save_training_rewards(learner=learner, path="reacher_training_rewards")
    #save_video(learner=learner, path=f"reacher_wall_{seed}_argmax_video", use_argmax=True)
    for i in range(2):
        save_video(learner=learner, path=f"reacher_wall_{seed}_random_video_{i}", use_argmax=False)
