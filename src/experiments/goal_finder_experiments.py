import numpy as np
import logging

from toy_environments.goal_finder import GoalFinderEnv
from experiments.scaffold import run_batch, ParamGrid
from experiments.noise import normal, uniform, adversarial, rescale_noise

logger = logging.basicConfig(level=logging.INFO)

# Run batch of experiments on goal finder environment
if __name__ == "__main__":
    run_batch(
        folder_path="../../results/goal_finder/",
        environment_type=GoalFinderEnv,
        param_grids=[
            ParamGrid(
                param_name="n_dimensions",
                grid=[2, 3, 4],
                is_environment_param=True
            ),
            ParamGrid(
                param_name="sparsity_param",
                grid=[2, 4, 8, 16],
                is_environment_param=True
            ),
            ParamGrid(
                param_name="reward_noise",
                grid=[0.0, 0.25, 0.5, 0.75, 1.0],
                is_environment_param=True
            ),
            ParamGrid(
                param_name="noise_sample",
                grid=[
                    normal,
                    rescale_noise(uniform, scaling_factor=1.0),
                    rescale_noise(adversarial, scaling_factor=1.0)
                ],
                is_environment_param=True
            ),
            ParamGrid(
                param_name="clipping_param",
                grid=[0.1, 0.2, 0.3, 0.4],
                is_environment_param=False
            ),
        ],
        default_hyperparams={
            "actor_hidden_layer_units": [64, 32],
            "critic_hidden_layer_units": [32, 18],
            "n_steps_per_trajectory": 16,
            "n_trajectories_per_batch": 64,
            "n_iterations": 200
        },
        n_cores=4,
        n_trials=5
    )
