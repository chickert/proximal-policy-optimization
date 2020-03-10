import logging

from algorithm.ppo import PPOLearner
from robot_environments.reacher import ReacherEnv
from utils.post_processing import save_video

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = ReacherEnv(render=False)

# Initialize learner
seed = 0
learner = PPOLearner(
    environment=environment,
    state_space_dimension=3,
    action_space_dimension=2,
    critic_hidden_layer_units=[32, 16],
    actor_hidden_layer_units=[64, 32],
    n_iterations=50,
    n_trajectories_per_batch=20,
    n_steps_per_trajectory=150,
    discrete_actor=False,
    seed=seed
)

# Train learner
learner.train()

# Save outputs
for i in range(2):
    save_video(learner=learner, path=f"reacher_{seed}_random_video_{i}", use_argmax=False)
