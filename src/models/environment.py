from typing import Callable, Tuple, Optional

import numpy as np


class Environment:

    def __init__(
            self,
            initial_state,
            transition_function: Callable,
            reward_function: Callable,
            is_done: Optional[Callable] = None
    ):
        self.initial_state = initial_state
        self.state = initial_state
        self.transition_function = transition_function
        self.reward_function = reward_function
        if is_done:
            self.is_done = is_done
        else:
            self.is_done = lambda state: False

    def update(self, action) -> Tuple[float, bool]:
        self.state = self.transition_function(state=self.state, action=action)
        done = self.is_done(self.state)
        reward = self.reward_function(state=self.state, action=action)
        return reward, done

    def reset(self):
        self.state = self.initial_state



