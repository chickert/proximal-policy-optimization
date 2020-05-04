import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset

from algorithms.ppo import PPOLearner
from algorithms.behavior_cloning import BCLearner
from environment_models.pusher import PusherEnv
from architectures.actor_critic import ActorCritic
from airobot_utils.pusher_simulator import PusherSimulator
import pybullet as p
import cv2


logger = logging.basicConfig(level=logging.DEBUG)

# Set paths
EXPERT_DATA_PATH = "../../data/expert.npz"
RESULTS_FOLDER = "../../data/results/pusher/"

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
        actor_std=1e-2
    )

    bc_learner = BCLearner(
        policy=policy,
        n_epochs=50,
        batch_size=128,
        learning_rate=3e-4
    )
    bc_learner.train(expert_data=expert_data)
    bc_learner.policy.save(path=f"{RESULTS_FOLDER}bc_model.pt")

    ppo_with_bc_learner = PPOLearner(
        environment=environment,
        policy=bc_learner.policy,
        n_steps_per_trajectory=32,
        n_trajectories_per_batch=64,
        n_epochs=5,
        n_iterations=20,
        learning_rate=3e-4,
        clipping_param=0.2,
        entropy_coefficient=1e-3,
        bc_coefficient=1e-3
    )
    trajectories = [actions for _, actions, _, _ in [ppo_with_bc_learner.generate_trajectory() for _ in range(1)]]

    simulator = PusherSimulator(render=True)
    simulator.render()
    for trajectory in trajectories:
        output = cv2.VideoWriter(f"test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for action in trajectory:
            simulator.apply_action(action)
            image = simulator.robot.cam.get_images(get_rgb=True, get_depth=False)[0]
            output.write(np.array(image))




    # ppo_with_bc_learner.environment.simulator = PusherSimulator(render=True)
    # p.connect(p.DIRECT)
    # for i in range(1):
    #     p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "test.mp4") #, f"{RESULTS_FOLDER}/bc_video_{i + 1}.mp4")
    #     _, actions, _, _ = ppo_with_bc_learner.generate_trajectory()
    #     p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

    # ppo_with_bc_learner.train(expert_data=expert_data, train_critic_only_on_init=True)
    # ppo_with_bc_learner.policy.save(path=f"{RESULTS_FOLDER}ppo_with_bc_model.pt")



