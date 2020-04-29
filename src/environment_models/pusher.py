
from environment_models.base import BaseEnv
from airobot_utils.pusher_simulator import PusherSimulator

import numpy as np


class PusherEnv(BaseEnv):

    def __init__(self):

        self.simulator = PusherSimulator(render=False)

        def transition_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> np.ndarray:
            self.simulator.apply_action(action)
            return self.simulator.get_obs()

        def reward_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> float:
            return self.simulator.compute_reward_push(state)

        BaseEnv.__init__(
            self,
            initial_state= self.simulator.get_obs(),
            transition_function=transition_function,
            reward_function=reward_function,
            state_space_dimension=9,
            action_space_dimension=2
        )

    def reset(self):
        self.simulator.reset()
