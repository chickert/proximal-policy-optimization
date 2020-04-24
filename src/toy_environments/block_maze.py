import numpy as np
from models.environment import Environment

from typing import Callable, Optional


def generate_maze(
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        maze_size: int,
        pct_blocked: float,
) -> np.ndarray:
    is_blocked = np.random.binomial(1, pct_blocked, (maze_size, maze_size))
    is_blocked[int(initial_state[0]), int(initial_state[1])] = 0
    is_blocked[int(goal_state[0]), int(goal_state[1])] = 0
    return is_blocked


class BlockMazeEnv(Environment):

    def __init__(
            self,
            initial_state: Optional[np.ndarray] = None,
            goal_state: Optional[np.ndarray] = None,
            maze_size: int = 10,
            pct_blocked: float = 0.1,
            sparsity_param: float = 1.0,
            reward_noise: float = 0,
            noise_sample: Callable = np.random.randn,
            rescale_sparsity_param: bool = True,
            seed: int = 0
    ):
        # Set-up initial and goal states
        if initial_state is None:
            initial_state = np.zeros(2)
        if goal_state is None:
            goal_state = np.array([maze_size - 1, maze_size - 1])
        assert len(initial_state) == len(goal_state) == 2, ValueError
        assert (max(*initial_state, *goal_state) < maze_size) and (min(*initial_state, *goal_state) >= 0), ValueError

        # Generate maze
        np.random.seed(seed)
        is_blocked = generate_maze(
            initial_state=initial_state,
            goal_state=goal_state,
            maze_size=maze_size,
            pct_blocked=pct_blocked
        )

        # Rescale sparsity param if desired
        if rescale_sparsity_param:
            sparsity_param = sparsity_param / np.sqrt(np.linalg.norm(initial_state - goal_state, 2))

        def transition_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> np.ndarray:
            x = max(min(state[0] + action[0], maze_size - 1), 0)
            y = max(min(state[1] + action[1], maze_size - 1), 0)
            if is_blocked[int(x), int(y)]:
                return state
            else:
                return np.array([x, y])

        def reward_function(
                state: np.ndarray,
                action: np.ndarray
        ) -> float:
            distance_to_goal = np.linalg.norm(state - goal_state, 2)
            return np.exp(-sparsity_param * distance_to_goal ** 2) + reward_noise*noise_sample()

        def is_done(state: np.ndarray) -> bool:
            if np.isclose(state, goal_state).all():
                return True
            else:
                return False

        Environment.__init__(
            self,
            initial_state=initial_state,
            transition_function=transition_function,
            reward_function=reward_function,
            is_done=is_done
        )
        self.goal_state = goal_state
