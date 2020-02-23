from robot_environments.reacher import ReacherEnv
from ppo_alogrithm import Actor, Critic, PPOLearner
import numpy as np

environment = ReacherEnv(render=False)
delta = 2
action_map = {
    0: delta*np.array([0, 0]),
    1: delta*np.array([1, 0]),
    2: delta*np.array([-1, 0]),
    3: delta*np.array([0, 1]),
    4: delta*np.array([0, -1]),
    5: delta*np.array([1, 1]),
    6: delta*np.array([-1, 1]),
    7: delta*np.array([1, -1]),
    8: delta*np.array([-1, -1]),
}

learner = PPOLearner(
    environment=environment,
    state_space_dimension=3,
    action_space_dimension=9,
    action_map=action_map,
    hidden_layer_size=10,
    discount=0.1,
    critic_discount=1e-3,
    batch_size=16
)


learner.train()

