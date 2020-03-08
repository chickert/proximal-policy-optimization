from robot_environments.reacher import ReacherEnv
from ppo_alogrithm import PPOLearner
import numpy as np
from utils.post_processing import save_training_rewards, save_videos
import logging

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = ReacherEnv(render=False)

# Define action map
step_size = 30.0
action_map = {
    0: step_size * np.array([0, 0]),
    1: step_size * np.array([1, 0]),
    2: step_size * np.array([-1, 0]),
    3: step_size * np.array([0, 1]),
    4: step_size * np.array([0, -1])
}


good_seeds = [0, 1, 8, 10, 12]
for seed in good_seeds:
    # Initialize learner
    learner = PPOLearner(
        environment=environment,
        state_space_dimension=3,
        action_space_dimension=5,
        action_map=action_map,
        critic_hidden_layer_units=[16, 8],
        actor_hidden_layer_units=[64, 32],
        seed=seed
    )

    # Train learner
    learner.train()

    # Save outputs
    save_training_rewards(learner=learner, path=f"reacher_training_rewards")
    #save_videos(learner=learner, path=f"reacher_{seed}_video")
