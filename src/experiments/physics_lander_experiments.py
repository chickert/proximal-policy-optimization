import numpy as np
import logging

from algorithm.annealing import AnnealedParam
from toy_environments.physics_lander import PhysicsLanderEnv, constant_force
from experiments.scaffold import run_batch, ParamGrid
from experiments.noise import normal, uniform, adversarial, rescale_noise


logger = logging.basicConfig(level=logging.DEBUG)

# Set experiment batch parameters
PATH = "../../results/physics_lander/"
N_CORES = 4
N_TRIALS = 5

# Set environment parameter grids
ENVIRONMENT_PARAM_GRIDS = [
    ParamGrid(
        param_name="force",
        grid=[None, constant_force(np.array([0, 1e-1]))],
    ),
    ParamGrid(
        param_name="sparsity_param",
        grid=[2, 4, 8],
    ),
    ParamGrid(
        param_name="reward_noise",
        grid=[0.0, 0.25, 0.5],
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
    "action_space_dimension": 2,
    "actor_hidden_layer_units": [64, 32],
    "critic_hidden_layer_units": [32, 18],
    "n_steps_per_trajectory": 32,
    "n_trajectories_per_batch": 64,
    "n_iterations": 100,
    "learning_rate": AnnealedParam(
        param_min=1e-4,
        param_max=5e-4,
        period=20,
        schedule_type="linear",
    )
}

# Run batch of experiments on goal finder environment
if __name__ == "__main__":
    run_batch(
        folder_path=PATH,
        environment_type=PhysicsLanderEnv,
        n_cores=N_CORES,
        n_trials=N_TRIALS,
        environment_param_grids=ENVIRONMENT_PARAM_GRIDS,
        ppo_param_grids=PPO_PARAM_GRIDS,
        fixed_ppo_params=FIXED_PPO_PARAMS
    )
