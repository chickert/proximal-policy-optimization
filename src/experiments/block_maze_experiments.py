import numpy as np
import logging

from algorithm.annealing import AnnealedParam
from toy_environments.block_maze import BlockMazeEnv
from experiments.scaffold import run_batch, ParamGrid
from experiments.noise import normal, uniform, adversarial, rescale_noise


logger = logging.basicConfig(level=logging.DEBUG)

# Set experiment batch parameters
PATH = "../../results/block_maze/"
N_CORES = 4
N_TRIALS = 5

# Set environment parameter grids
ENVIRONMENT_PARAM_GRIDS = [
    ParamGrid(
        param_name="maze_size",
        grid=[5, 10, 20],
    ),
    ParamGrid(
        param_name="pct_blocked",
        grid=[0.05, 0.1, 0.2],
    ),
    ParamGrid(
        param_name="sparsity_param",
        grid=[2, 4, 8, 16],
    ),
    ParamGrid(
        param_name="reward_noise",
        grid=[0.0, 0.25, 0.5, 0.75, 1.0],
    ),
    ParamGrid(
        param_name="noise_sample",
        grid=[
            normal,
            rescale_noise(uniform, scaling_factor=1.0),
            rescale_noise(adversarial, scaling_factor=1.0)
        ],
    )
]

# Set PPO parameter grids
PPO_PARAM_GRIDS = [
    ParamGrid(
        param_name="clipping_param",
        grid=[0.1, 0.2, 0.3, 0.4, AnnealedParam(param_min=0.1, param_max=0.4, period=20)]
    ),
    ParamGrid(
        param_name="clipping_type",
        grid=["clamp", "sigmoid", "tanh"]
    ),
]

# Set fixed PPO parameters
FIXED_PPO_PARAMS = {
    "action_space_dimension": 4,
    "actor_hidden_layer_units": [64, 32],
    "critic_hidden_layer_units": [32, 18],
    "n_steps_per_trajectory": 32,
    "n_trajectories_per_batch": 128,
    "n_iterations": 100,
    "learning_rate": AnnealedParam(
        param_min=1e-4,
        param_max=2e-4,
        period=20,
        schedule_type="linear",
    ),
    "action_map": {
        0: np.array([1, 0]),
        1: np.array([-1, 0]),
        2: np.array([0, 1]),
        3: np.array([0, -1])
    }
}

# Run batch of experiments on goal finder environment
if __name__ == "__main__":
    run_batch(
        folder_path=PATH,
        environment_type=BlockMazeEnv,
        n_cores=N_CORES,
        n_trials=N_TRIALS,
        environment_param_grids=ENVIRONMENT_PARAM_GRIDS,
        ppo_param_grids=PPO_PARAM_GRIDS,
        fixed_ppo_params=FIXED_PPO_PARAMS
    )
