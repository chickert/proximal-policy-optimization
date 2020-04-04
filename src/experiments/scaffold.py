from typing import Optional, Dict, Type, Iterable, List, Any

import numpy as np
import logging
from pathos.multiprocessing import Pool, cpu_count

from algorithm.ppo import PPOLearner
from models.environment import Environment
from utils.misc import combine_grids

# Set up logging
logger = logging.getLogger(__name__)


def run_experiment(
        environment: Environment,
        training_rewards_path: str,
        pool: Optional[Pool] = None,
        n_trials: int = 5,
        ppo_params: Optional[Dict[str, Any]] = None,
        action_map: Optional[Dict[int, np.ndarray]] = None
) -> None:

    for seed in range(n_trials):

        # Initialize learner
        if action_map:
            learner = PPOLearner(
                environment=environment,
                action_map=action_map,
                seed=seed,
                **ppo_params
            )
        else:
            learner = PPOLearner(
                environment=environment,
                seed=seed,
                **ppo_params
            )

        # Train learner
        learner.train(pool=pool)

        # Save learner outputs
        learner.save_training_rewards(path=f"{training_rewards_path}.csv")


class ParamGrid:

    def __init__(
            self,
            param_name: str,
            grid: Iterable
    ):
        self.param_name = param_name
        self.grid = grid


def make_experiment_id(
        environment_params: Dict[str, Any],
        ppo_params: Dict[str, Any],
) -> str:
    param_strings = []
    for params in (environment_params, ppo_params):
        for key, value in params.items():
            if type(value) is float:
                param_strings.append(f"{key}_{'{0:.2f}'.format(value)}".replace(".", "-"))
            elif callable(value):
                param_strings.append(f"{key}_{value.__name__}")
            else:
                param_strings.append(f"{key}_{value}")
    return "_".join(param_strings)


def run_batch(
        folder_path: str,
        environment_type: Type[Environment],
        environment_param_grids: List[ParamGrid],
        ppo_param_grids: List[ParamGrid],
        fixed_ppo_params: Optional[Dict[str, Any]] = None,
        n_trials: int = 5,
        n_cores: Optional[int] = None,
        action_map: Optional[Dict[int, np.ndarray]] = None
) -> None:

    # Set up multiprocessing pool
    if n_cores is None:
        n_cores = cpu_count()
    pool = Pool(n_cores)

    # Run experiment for all environment parameter and hyper parameter combinations
    for environment_params in combine_grids(
        *zip(*[(grid.grid, grid.param_name) for grid in environment_param_grids])
    ):
        environment = environment_type(**environment_params)
        for ppo_params in combine_grids(
            *zip(*[(grid.grid, grid.param_name) for grid in ppo_param_grids])
        ):
            experiment_id = make_experiment_id(
                environment_params=environment_params,
                ppo_params=ppo_params
            )
            if fixed_ppo_params:
                ppo_params = dict(**ppo_params, **fixed_ppo_params)
            run_experiment(
                environment=environment,
                training_rewards_path=f"{folder_path}{experiment_id}",
                pool=pool,
                ppo_params=ppo_params,
                action_map=action_map,
                n_trials=n_trials
            )

    # Close pool
    pool.close()
