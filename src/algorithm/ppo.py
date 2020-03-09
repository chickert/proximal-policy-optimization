import logging
from copy import deepcopy
from typing import Tuple, Union, Dict, List, Optional

import numpy as np
import torch
from gym.spaces import Box

from algorithm.actor_critic import ActorCritic
from robot_environments.pusher import PusherEnv
from robot_environments.reacher import ReacherEnv
from robot_environments.reacher_wall import ReacherWallEnv
from utils.misc import concatenate_lists, timer

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logger = logging.getLogger(__name__)

# Constants
TOL = 1e-10


class PPOLearner:

    def __init__(
            self,
            environment: Union[ReacherEnv, ReacherWallEnv, PusherEnv],
            state_space_dimension: int,
            action_space_dimension: int,
            critic_hidden_layer_units: List[int],
            actor_hidden_layer_units: List[int],
            discrete_actor: bool = False,
            random_init_box: Optional[Box] = None,
            n_steps_per_trajectory: int = 200,
            n_trajectories_per_batch: int = 10,
            n_epochs: int = 4,
            n_iterations: int = 50,
            learning_rate: float = 3e-4,
            discount: float = 0.99,
            clipping_param: float = 0.2,
            critic_coefficient: float = 1.0,
            init_entropy_coefficient: float = 0.01,
            entropy_decay: float = 0.999,
            init_reversion_threshold: float = 0.4,
            reversion_threshold_decay: float = 0.95,
            min_reversion_threshold: float = 0.1,
            allow_policy_reversions: bool = False,
            seed: int = 0
    ):
        # Set seed
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set environment attribute
        self.environment = environment

        # Initialize actor-critic policy network
        self.policy = ActorCritic(
            state_space_dimension=state_space_dimension,
            action_space_dimension=action_space_dimension,
            actor_hidden_layer_units=actor_hidden_layer_units,
            critic_hidden_layer_units=critic_hidden_layer_units,
            discrete_actor=discrete_actor
        )

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Set hyper-parameter attributes
        self.random_init_box = random_init_box
        self.n_steps_per_trajectory = n_steps_per_trajectory
        self.n_trajectories_per_batch = n_trajectories_per_batch
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.discount = discount
        self.clipping_param = clipping_param
        self.critic_coefficient = critic_coefficient
        self.entropy_coefficient = init_entropy_coefficient
        self.entropy_decay = entropy_decay
        self.reversion_threshold = init_reversion_threshold
        self.reversion_threshold_decay = reversion_threshold_decay
        self.min_reversion_threshold = min_reversion_threshold
        self.allow_policy_reversions = allow_policy_reversions
        if random_init_box is None:
            self.randomize_init_state = False
        else:
            self.randomize_init_state = True

        # Initialize attributes for performance tracking
        self.mean_rewards = []
        self.best_mean_reward = -np.inf
        self.best_policy = deepcopy(self.policy)

    def calculate_discounted_rewards(self, rewards: List[float]) -> List[float]:
        discounted_rewards = []
        discounted_reward = 0
        for t in reversed(range(self.n_steps_per_trajectory)):
            discounted_reward = rewards[t] + self.discount * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        return discounted_rewards

    def generate_sample_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:

        states = []
        actions = []
        rewards = []

        # Set initial state
        if self.randomize_init_state:
            nominal_init = self.environment.init
            self.environment.init = self.random_init_box.sample()
            self.environment.reset()
        state = self.environment._get_obs()

        # Generate a sample trajectory under the current policy
        for _ in range(self.n_steps_per_trajectory + 1):
            # Sample from policy and receive feedback from environment
            action = self.policy.sample_action(state)
            new_state, reward, done, info = self.environment.step(action)
            #print(action)

            # Store information from step
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Update state
            state = new_state

        # Reset environment to initial state
        if self.randomize_init_state:
            self.environment.init = nominal_init
        self.environment.reset()

        # Calculate discounted rewards
        discounted_rewards = self.calculate_discounted_rewards(rewards=rewards)

        # Return states (excluding terminal state), actions, rewards and discounted rewards
        return states[:-1], actions[:-1], rewards, discounted_rewards

    def generate_argmax_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:

        states = []
        actions = []
        rewards = []

        # Set initial state
        state = self.environment._get_obs()

        # Generate trajectory corresponding to argmax of probabilities under current policy
        for _ in range(self.n_steps_per_trajectory + 1):
            # Get argmax action from policy and receive feedback from environment
            action = self.policy.get_argmax_action(state)
            new_state, reward, done, info = self.environment.step(action)

            # Store information from step
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Update state
            state = new_state

        # Reset environment to initial state
        self.environment.reset()

        # Calculate discounted rewards
        discounted_rewards = self.calculate_discounted_rewards(rewards=rewards)

        # Return states (excluding terminal state), actions, rewards and discounted rewards
        return states[:-1], actions[:-1], rewards, discounted_rewards

    @timer
    def generate_batch(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:

        # Generate trajectories
        trajectories = [self.generate_sample_trajectory() for _ in range(self.n_trajectories_per_batch)]

        # Unzip and return trajectories
        states, actions, rewards, discounted_rewards = map(concatenate_lists, zip(*trajectories))
        return states, actions, rewards, discounted_rewards

    def track_performance(self, rewards):
        mean_reward = np.mean(rewards)
        self.mean_rewards.append(mean_reward)
        logger.info(f"Mean reward: {mean_reward}")
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.best_policy = deepcopy(self.policy)
        elif ((self.best_mean_reward - mean_reward) / self.best_mean_reward > self.reversion_threshold) \
                & self.allow_policy_reversions:
            self.policy = self.best_policy
            logger.info(f"Reverting policy!")

    def get_tensors(
            self,
            states: List[np.ndarray],
            actions: List[np.ndarray],
            discounted_rewards: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Convert data to tensors
        states = torch.tensor(states).float().to(device).detach()
        actions = torch.tensor(actions).float().to(device).detach()
        discounted_rewards = torch.tensor(discounted_rewards).float().unsqueeze(1).to(device).detach()
        old_log_probabilities = self.policy.get_distribution(states).log_prob(actions).float().to(device).detach()

        # Normalize discounted rewards
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (
                    torch.std(discounted_rewards) + TOL)

        return states, actions, discounted_rewards, old_log_probabilities

    def ppo_loss(
            self,
            values: torch.Tensor,
            discounted_rewards: torch.Tensor,
            log_probabilities: torch.Tensor,
            old_log_probabilities: torch.Tensor,
            entropy: torch.Tensor,
            advantage_estimates: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        actor_loss = -torch.mean(
            torch.min(
                torch.clamp(ratio, 1 - self.clipping_param, 1 + self.clipping_param) * advantage_estimates,
                ratio * advantage_estimates
            )
        )
        critic_loss = torch.mean((discounted_rewards - values).pow(2))
        entropy_loss = -torch.mean(entropy)
        return actor_loss + self.critic_coefficient * critic_loss + self.entropy_coefficient * entropy_loss

    @timer
    def update_policy_network(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            discounted_rewards: torch.Tensor,
            old_log_probabilities: torch.Tensor
    ) -> None:

        for _ in range(self.n_epochs):

            # Get policy network outputs
            log_probabilities, values, entropy = self.policy(
                states=states,
                actions=actions
            )

            # Get advantage estimates
            advantage_estimates = discounted_rewards - values.detach()

            # Calculate loss
            loss = self.ppo_loss(
                values=values,
                discounted_rewards=discounted_rewards,
                log_probabilities=log_probabilities,
                old_log_probabilities=old_log_probabilities,
                entropy=entropy,
                advantage_estimates=advantage_estimates
            )

            # Perform gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_hyperparameters(self) -> None:

        # Decay entropy coefficient
        self.entropy_coefficient = self.entropy_decay * self.entropy_coefficient

        # Decay reversion threshold
        self.reversion_threshold = max(self.reversion_threshold_decay * self.reversion_threshold,
                                       self.min_reversion_threshold)

    def train(self) -> None:

        for i in range(self.n_iterations):

            logger.info(f"Iteration: {i + 1}")

            # Generate batch
            states, actions, rewards, discounted_rewards = self.generate_batch()

            # Track performance
            self.track_performance(rewards=rewards)

            # Convert data to PyTorch tensors
            states, actions, discounted_rewards, old_log_probabilities = self.get_tensors(
                states=states,
                actions=actions,
                discounted_rewards=discounted_rewards
            )

            # Perform gradient updates
            self.update_policy_network(
                states=states,
                actions=actions,
                discounted_rewards=discounted_rewards,
                old_log_probabilities=old_log_probabilities
            )

            # Update hyper-parameters
            self.update_hyperparameters()

            logger.info("-" * 50)


