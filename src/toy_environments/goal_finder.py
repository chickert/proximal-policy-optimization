import numpy as np
from models.environment import Environment


class GoalFinderEnv(Environment):

    def __init__(
            self,
            initial_state: np.ndarray,
            goal_state: np.ndarray,
            sparsity_param: float,
            reward_noise: float = 0
    ):
        def transition_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> np.ndarray:
            return state + action

        def reward_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> float:
            distance_to_goal = np.linalg.norm(state - goal_state, 2)
            return np.exp(-sparsity_param * distance_to_goal ** 2) + reward_noise*np.random.randn()

        Environment.__init__(
            self,
            initial_state=initial_state,
            transition_function=transition_function,
            reward_function=reward_function
        )
        self.goal_state = goal_state
