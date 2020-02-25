from robot_environments.reacher import ReacherEnv
from ppo_alogrithm import Actor, Critic, PPOLearner
import numpy as np

environment = ReacherEnv(render=False)
delta = 3
action_map = {
    0: delta*np.array([0, 0]),
    1: delta*np.array([1, 0]),
    2: delta*np.array([-1, 0]),
    3: delta*np.array([0, 1]),
    4: delta*np.array([0, -1])
}

learner = PPOLearner(
    environment=environment,
    state_space_dimension=3,
    action_space_dimension=5,
    action_map=action_map,
    critic_hidden_layer_units=[32],
    actor_hidden_layer_units=[32],
    horizon=512,
    discount=0.9,
    critic_discount=1e-3,
    entropy_coefficient=1e-1,
    batch_size=64,
    learning_rate=1e-4
)


learner.train()

