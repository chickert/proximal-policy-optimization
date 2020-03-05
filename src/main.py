from robot_environments.reacher import ReacherEnv
from ppo_alogrithm import PPOLearner
from gym.spaces import Box
import numpy as np


environment = ReacherEnv(render=False)
step_size = 2
action_map = {
    0: step_size * np.array([0, 0]),
    1: step_size * np.array([1, 0]),
    2: step_size * np.array([-1, 0]),
    3: step_size * np.array([0, 1]),
    4: step_size * np.array([0, -1])
}

learner = PPOLearner(
    environment=environment,
    state_space_dimension=3,
    action_space_dimension=5,
    action_map=action_map,
    critic_hidden_layer_units=[32, 16],
    actor_hidden_layer_units=[32, 16],
    random_init_box=Box(
        high=np.array([0.55, 0.15, 1.0]),
        low=np.array([0.45, 0.05, 1.0]),
        dtype=np.float64
    )
)


learner.train()

