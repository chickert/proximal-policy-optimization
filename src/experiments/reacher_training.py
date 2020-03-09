import logging

from algorithm.ppo import PPOLearner
from robot_environments.reacher import ReacherEnv
from utils.post_processing import save_video

logger = logging.basicConfig(level=logging.INFO)

# Set environment
environment = ReacherEnv(render=False)

# Run training over multiple random seeds
for seed in [2]:

    # Initialize learner
    learner = PPOLearner(
        environment=environment,
        state_space_dimension=3,
        action_space_dimension=2,
        critic_hidden_layer_units=[32, 16],
        actor_hidden_layer_units=[64, 32],
        n_iterations=50,
        n_trajectories_per_batch=10,
        n_steps_per_trajectory=200,
        discrete_actor=True,
        seed=seed
    )

    # Train learner
    learner.train()

    # Save outputs
    #save_training_rewards(learner=learner, path="reacher_training_rewards")
    #save_video(learner=learner, path=f"reacher_{seed}_argmax_video", use_argmax=True)
    for i in range(2):
        save_video(learner=learner, path=f"reacher_{seed}_random_video_{i}", use_argmax=False)
