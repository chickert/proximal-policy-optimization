import numpy as np

from environment_models.base import BaseEnv

from typing import Callable, Optional


class PhysicsLanderEnv(BaseEnv):

    def __init__(
            self,
            initial_position: np.ndarray = np.ones(2),
            goal_position: np.ndarray = np.zeros(2),
            sparsity_param: float = 1.0,
            penalty: float = 0.0,
            reward_noise: float = 0.0,
            noise_sample: Callable = np.random.randn,
            force: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            rescale_sparsity_param: bool = True,
            seed: int = 0
    ):

        n_dimensions = initial_position.shape[0]
        initial_state = np.concatenate((initial_position, np.zeros(n_dimensions)), axis=0)
        if rescale_sparsity_param:
            sparsity_param = sparsity_param / np.linalg.norm(initial_position - goal_position, 2)

        if force is None:
            def force(position: np.ndarray) -> np.ndarray:
                return np.zeros(n_dimensions)

        def transition_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> np.ndarray:
            position = state[:n_dimensions]
            velocity = state[n_dimensions:]
            position = position + velocity + (action + force(position))/2
            velocity = velocity + action
            return np.concatenate((position, velocity), axis=0)

        def reward_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> float:
            distance_to_goal = np.linalg.norm(state[:n_dimensions] - goal_position, 2)
            cost = np.linalg.norm(action, 2)
            return np.exp(-sparsity_param*distance_to_goal**2 - penalty*cost**2) + reward_noise*noise_sample()

        BaseEnv.__init__(
            self,
            initial_state=initial_state,
            transition_function=transition_function,
            reward_function=reward_function,
            action_space_dimension=int(len(initial_state) / 2)
        )

        np.random.seed(seed)
        self.goal_position = goal_position


def constant_force(direction: np.ndarray) -> Callable:
    def force(position: np.ndarray) -> np.ndarray:
        return direction
    force.__name__ = "constant_force"
    return force
