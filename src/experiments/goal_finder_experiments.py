import logging

from algorithms.param_annealing import AnnealedParam
from environment_models.goal_finder import GoalFinderEnv
from experiments.scaffold import run_batch, ParamGrid
from experiments.random_noise_generation import normal, uniform, adversarial, rescale_noise


logger = logging.basicConfig(level=logging.DEBUG)

# Set experiment batch parameters
PATH = "../../data/results/goal_finder/"
N_CORES = 4
N_TRIALS = 5

# Set environment parameter grids
ENVIRONMENT_PARAM_GRIDS = [
    ParamGrid(
        param_name="n_dimensions",
        grid=[2, 3],
    ),
    ParamGrid(
        param_name="sparsity_param",
        grid=[2, 4, 8],
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
        grid=[0.2, 0.3, 0.4, AnnealedParam(param_min=0.1, param_max=0.4, period=20)]
    ),
    ParamGrid(
        param_name="clipping_type",
        grid=["clamp", "sigmoid", "tanh"]
    )
]

# Set fixed PPO parameters
FIXED_PPO_PARAMS = {
    "actor_hidden_layer_units": [128, 64],
    "critic_hidden_layer_units": [64, 32],
    "n_steps_per_trajectory": 16,
    "n_trajectories_per_batch": 64,
    "n_iterations": 100,
    "learning_rate": AnnealedParam(
        param_min=2e-4,
        param_max=4e-4,
        period=20,
        schedule_type="linear",
    )
}

# Run batch of experiments on goal finder environment
if __name__ == "__main__":
    run_batch(
        folder_path=PATH,
        environment_type=GoalFinderEnv,
        n_cores=N_CORES,
        n_trials=N_TRIALS,
        environment_param_grids=ENVIRONMENT_PARAM_GRIDS,
        ppo_param_grids=PPO_PARAM_GRIDS,
        fixed_ppo_params=FIXED_PPO_PARAMS
    )
