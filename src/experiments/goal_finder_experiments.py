import numpy as np
import logging

from algorithm.ppo import PPOLearner
from toy_environments.goal_finder import GoalFinderEnv

logger = logging.basicConfig(level=logging.INFO)

# Constants
N_DIMENSIONS = 2
DISCRETE_STEP_SIZE = 5e-2
SPARSITY_PARAM = 0.25

# Initial environment
environment = GoalFinderEnv(
    initial_state=np.zeros(N_DIMENSIONS),
    goal_state=np.ones(N_DIMENSIONS),
    sparsity_param=SPARSITY_PARAM
)

# Action map
action_map = {
    key: DISCRETE_STEP_SIZE * action for key, action in enumerate(
        [np.identity(N_DIMENSIONS)[i] for i in range(N_DIMENSIONS)]
        + [-np.identity(N_DIMENSIONS)[i] for i in range(N_DIMENSIONS)]
        + [np.zeros(N_DIMENSIONS)]
    )
}

# Initialize learners
discrete_ppo_leaner = PPOLearner(
    environment=environment,
    state_space_dimension=N_DIMENSIONS,
    action_space_dimension=len(action_map),
    actor_hidden_layer_units=[64, 32],
    critic_hidden_layer_units=[32, 16],
    discrete_actor=True,
    action_map=action_map
)
continuous_ppo_learner = PPOLearner(
    environment=environment,
    state_space_dimension=N_DIMENSIONS,
    action_space_dimension=N_DIMENSIONS,
    actor_hidden_layer_units=[64, 32],
    critic_hidden_layer_units=[32, 16],
    discrete_actor=False,
    actor_std=0.5,
    seed=1
)

# Train learners
#discrete_ppo_leaner.train()

continuous_ppo_learner.train()
