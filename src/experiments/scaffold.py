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
        hyperparams: Optional[Dict[str, Any]] = None,
        action_map: Optional[Dict[int, np.ndarray]] = None
) -> None:

    for seed in range(n_trials):

        # Initialize learner
        if action_map:
            learner = PPOLearner(
                environment=environment,
                discrete_actor=True,
                action_map=action_map,
                seed=seed,
                **hyperparams
            )
        else:
            learner = PPOLearner(
                environment=environment,
                seed=seed,
                **hyperparams
            )

        # Train learner
        learner.train(pool=pool)

        # Save learner outputs
        learner.save_training_rewards(path=f"{training_rewards_path}.csv")


class ParamGrid:

    def __init__(
            self,
            param_name: str,
            grid: Iterable,
            is_environment_param: bool = False,
    ):
        self.param_name = param_name
        self.grid = grid
        self.is_environment_param = is_environment_param


def make_experiment_id(
        environment_params: Dict[str, Any],
        hyperparams: Dict[str, Any],
) -> str:
    param_strings = []
    for params in (environment_params, hyperparams):
        for key, value in params.items():
            if type(value) is float:
                param_strings.append(f"{key}_{'{0:.2f}'.format(value)}".replace(".", "-"))
            elif type(value) is int:
                param_strings.append(f"{key}_{value}")
            elif callable(value):
                param_strings.append(f"{key}_{value.__name__}")
            else:
                print(key, value)
                raise NotImplemented
    return "_".join(param_strings)


def run_batch(
        folder_path: str,
        environment_type: Type[Environment],
        param_grids: List[ParamGrid],
        default_hyperparams: Optional[Dict[str, Any]] = None,
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
        *zip(*[(grid.grid, grid.param_name) for grid in param_grids if grid.is_environment_param])
    ):
        environment = environment_type(**environment_params)
        for hyperparams in combine_grids(
            *zip(*[(grid.grid, grid.param_name) for grid in param_grids if not grid.is_environment_param])
        ):
            experiment_id = make_experiment_id(
                environment_params=environment_params,
                hyperparams=hyperparams
            )
            if default_hyperparams:
                hyperparams = dict(**hyperparams, **default_hyperparams)
            run_experiment(
                environment=environment,
                training_rewards_path=f"{folder_path}{experiment_id}",
                pool=pool,
                hyperparams=hyperparams,
                action_map=action_map,
                n_trials=n_trials
            )

    # Close pool
    pool.close()
