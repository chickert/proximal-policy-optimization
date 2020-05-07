import logging
from copy import deepcopy
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from airobot_utils.pusher_simulator import PusherSimulator
from algorithms.behavior_cloning import BCLearner
from algorithms.ppo import PPOLearner
from architectures.actor_critic import ActorCritic
from environment_models.pusher import PusherEnv

logger = logging.basicConfig(level=logging.DEBUG)

# Set paths
EXPERT_DATA_PATH = "../../data/expert.npz"
RESULTS_FOLDER = "../../data/results/pusher/"

# Constants
TRAIN_BC = False
PPO_PARAMS = dict(
    n_steps_per_trajectory=64,
    n_trajectories_per_batch=16,
    n_epochs=5,
    n_iterations=50,
    learning_rate=3e-4,
    clipping_param=0.2,
    entropy_coefficient=1e-3
)



def run_trials(learner: PPOLearner, n_trials: int = 100) -> Tuple[List[List[np.ndarray]], List[float]]:
    trajectories = []
    errors = []
    for i in range(n_trials):
        _, actions, _, _ = learner.generate_trajectory(use_argmax=True, perform_reset=False)
        trajectories.append(actions)
        state = learner.environment.simulator.get_obs()
        errors.append(np.linalg.norm(state[3:6] - state[6:9]))
        learner.environment.reset()
    return trajectories, errors


def save_video(trajectories: List[List[np.ndarray]], simulator: PusherSimulator, path: str) -> None:
        output = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for actions in trajectories:
            for action in actions:
                simulator.apply_action(action)
                image = simulator.robot.cam.get_images(get_rgb=True, get_depth=False)[0]
                output.write(np.array(image))
            simulator.reset()


if __name__ == "__main__":

    # Load data
    expert_data = np.load(EXPERT_DATA_PATH)
    expert_data = TensorDataset(torch.tensor(expert_data["obs"]), torch.tensor(expert_data["action"]))

    environment = PusherEnv()

    policy = ActorCritic(
        state_space_dimension=environment.state_space_dimension,
        action_space_dimension=environment.action_space_dimension,
        actor_hidden_layer_units=(128, 64),
        critic_hidden_layer_units=(64, 32),
        actor_std=5e-2,
        activation=nn.Tanh
    )

    if TRAIN_BC:

        bc_learner = BCLearner(
            policy=deepcopy(policy),
            n_epochs=50,
            batch_size=128,
            learning_rate=3e-4
        )
        bc_learner.train(expert_data=expert_data)
        pd.DataFrame(bc_learner.training_loss, columns=["loss"]).to_csv(f"{RESULTS_FOLDER}bc_training_loss.csv")
        bc_learner.policy.save(path=f"{RESULTS_FOLDER}bc_model.pt")

        learner = PPOLearner(
            environment=environment,
            policy=deepcopy(bc_learner.policy),
            n_steps_per_trajectory=64
        )

        trajectories, errors = run_trials(learner=learner)
        pd.DataFrame(
            {
                "mean": np.mean(errors),
                "standard_error": np.std(errors) / np.sqrt(len(errors)),
                "min": np.min(errors),
                "max": np.max(errors)
            },
            index=["behavioral_cloning"]
        ).to_csv(f"{RESULTS_FOLDER}bc_results.csv")

        simulator = PusherSimulator(render=True)
        simulator.render()
        save_video(trajectories=trajectories[:10], simulator=simulator, path=f"{RESULTS_FOLDER}bc_video.mp4")

    else:

        tabula_rasa_ppo_leaner = PPOLearner(
            environment=environment,
            policy=deepcopy(policy),
            bc_coefficient=0,
            **PPO_PARAMS
        )

        policy.load(path=f"{RESULTS_FOLDER}bc_model.pt")

        fine_tuned_ppo_learner = PPOLearner(
            environment=environment,
            policy=deepcopy(policy),
            bc_coefficient=0,
            **PPO_PARAMS
        )

        joint_loss_learner = PPOLearner(
            environment=environment,
            policy=deepcopy(policy),
            bc_coefficient=0.5,
            **PPO_PARAMS
        )

        tabula_rasa_ppo_leaner.train()
        fine_tuned_ppo_learner.train(train_critic_only_on_init=True)
        joint_loss_learner.train(expert_data=expert_data, train_critic_only_on_init=True)

        learner_names = ["tabula_rasa", "fine_tuned", "joint_loss"]
        learners = [tabula_rasa_ppo_leaner, fine_tuned_ppo_learner, joint_loss_learner]
        results = []
        saved_trajectories = {}
        for name, learner in zip(learner_names, learners):
            learner.save_training_rewards(f"{RESULTS_FOLDER}{name}_rewards.csv")
            trajectories, errors = run_trials(learner=learner)
            results.append(
                {
                    "mean": np.mean(errors),
                    "standard_error": np.std(errors) / np.sqrt(len(errors)),
                    "min": np.min(errors),
                    "max": np.max(errors)
                },
            )
            saved_trajectories[name] = trajectories[:10]
        pd.DataFrame(results, index=learner_names).to_csv(f"{RESULTS_FOLDER}ppo_results.csv")

        simulator = PusherSimulator(render=True)
        simulator.render()
        for name in learner_names:
            save_video(trajectories=saved_trajectories[name], simulator=simulator, path=f"{RESULTS_FOLDER}{name}_video.mp4")








