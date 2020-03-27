import logging
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import pathos.multiprocessing as mp

from models.actor_critic import ActorCritic
from models.environment import Environment
from utils.misc import concatenate_lists, timer

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logger = logging.getLogger(__name__)


class PPOLearner:

    def __init__(
            self,
            environment: Environment,
            state_space_dimension: int,
            action_space_dimension: int,
            critic_hidden_layer_units: List[int],
            actor_hidden_layer_units: List[int],
            discrete_actor: bool = False,
            action_map: Optional[Dict[int, np.ndarray]] = None,
            n_steps_per_trajectory: int = 32,
            n_trajectories_per_batch: int = 128,
            n_epochs: int = 4,
            n_iterations: int = 200,
            learning_rate: float = 3e-4,
            discount: float = 0.99,
            clipping_param: float = 0.2,
            critic_coefficient: float = 1.0,
            entropy_coefficient: float = 0.01,
            entropy_decay: float = 1.0,
            actor_std: float = 0.05,
            actor_std_decay: float = 0.999,
            parallelize_batch_generation: bool = True,
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
            discrete_actor=discrete_actor,
            action_map=action_map,
            actor_std=actor_std
        )

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Set hyper-parameter attributes
        self.n_steps_per_trajectory = n_steps_per_trajectory
        self.n_trajectories_per_batch = n_trajectories_per_batch
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.discount = discount
        self.clipping_param = clipping_param
        self.critic_coefficient = critic_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.entropy_decay = entropy_decay
        self.actor_std_decay = actor_std_decay

        # Initialize attributes for performance tracking
        self.mean_rewards = []
        self.mean_discounted_returns = []

        # Set other attributes
        self.parallelize_batch_generation = parallelize_batch_generation

    def calculate_discounted_returns(self, states: List[np.ndarray], rewards: List[float]) -> List[float]:
        discounted_returns = []
        discounted_return = 0  # TODO: check if this is the correct terminal condition
        for t in reversed(range(self.n_steps_per_trajectory)):
            discounted_return = rewards[t] + self.discount * discounted_return
            discounted_returns.insert(0, discounted_return)
        return discounted_returns

    def generate_trajectory(
            self,
            use_argmax: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:

        states = []
        actions = []
        rewards = []

        # Generate a trajectory under the current policy
        for _ in range(self.n_steps_per_trajectory + 1):

            # Sample from policy and receive feedback from environment
            if use_argmax:
                action = self.policy.get_argmax_action(self.environment.state)
            else:
                action = self.policy.sample_action(self.environment.state)

            # Store state and action
            states.append(self.environment.state)
            actions.append(action)

            # Perform update
            reward, done = self.environment.update(action)
            rewards.append(reward)
            if done:
                break

        # Reset environment to initial state
        self.environment.reset()

        # Calculate discounted rewards
        discounted_returns = self.calculate_discounted_returns(states=states, rewards=rewards)

        # Return states (excluding terminal state), actions, rewards and discounted rewards
        return states[:-1], actions[:-1], rewards, discounted_returns


    @timer
    def generate_batch(
            self,
            pool: mp.Pool
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:

        # Generate batch of trajectories
        if self.parallelize_batch_generation:
            trajectories = pool.starmap(self.generate_trajectory, [() for _ in range(self.n_trajectories_per_batch)])
        else:
            trajectories = [self.generate_trajectory() for _ in range(self.n_trajectories_per_batch)]

        # Unzip and return trajectories
        states, actions, rewards, discounted_returns = map(concatenate_lists, zip(*trajectories))
        return states, actions, rewards, discounted_returns

    def get_tensors(
            self,
            states: List[np.ndarray],
            actions: List[np.ndarray],
            discounted_returns: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Convert data to tensors
        states = torch.tensor(states).float().to(device).detach()
        if self.policy.discrete_actor:
            actions = [self.policy.inverse_action_map[tuple(action)] for action in actions]
        actions = torch.tensor(actions).float().to(device).detach()
        discounted_returns = torch.tensor(discounted_returns).float().unsqueeze(1).to(device).detach()
        old_log_probabilities = self.policy.get_distribution(states).log_prob(actions).float().to(device).detach()

        # Normalize discounted rewards
        discounted_returns = (discounted_returns - torch.mean(discounted_returns)) / (
                    torch.std(discounted_returns) + 1e-5)

        return states, actions, discounted_returns, old_log_probabilities

    def ppo_loss(
            self,
            values: torch.Tensor,
            discounted_returns: torch.Tensor,
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
        critic_loss = torch.mean((discounted_returns - values).pow(2))
        entropy_loss = -torch.mean(entropy)
        return actor_loss + self.critic_coefficient*critic_loss + self.entropy_coefficient*entropy_loss

    @timer
    def update_policy_network(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            discounted_returns: torch.Tensor,
            old_log_probabilities: torch.Tensor
    ) -> None:

        for _ in range(self.n_epochs):

            # Get policy network outputs
            log_probabilities, values, entropy = self.policy(
                states=states,
                actions=actions
            )

            # Get advantage estimates
            advantage_estimates = discounted_returns - values.detach()

            # Calculate loss
            loss = self.ppo_loss(
                values=values,
                discounted_returns=discounted_returns,
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

        # Decay actor standard deviation
        if not self.policy.discrete_actor:
            self.policy.actor_std = self.actor_std_decay * self.policy.actor_std

        # Do other stuff in the future

    @timer
    def train(self) -> None:

        pool = mp.Pool(mp.cpu_count())

        for i in range(self.n_iterations):

            logger.info(f"Iteration: {i + 1}")

            # Generate batch
            states, actions, rewards, discounted_returns = self.generate_batch(pool=pool)

            # Track performance
            self.mean_rewards.append(np.mean(rewards))
            self.mean_discounted_returns.append(np.mean(discounted_returns))

            # Convert data to PyTorch tensors
            states, actions, discounted_returns, old_log_probabilities = self.get_tensors(
                states=states,
                actions=actions,
                discounted_returns=discounted_returns
            )

            # Perform gradient updates
            self.update_policy_network(
                states=states,
                actions=actions,
                discounted_returns=discounted_returns,
                old_log_probabilities=old_log_probabilities
            )

            # Update hyper-parameters
            self.update_hyperparameters()

            # Log performance
            logger.info(f"Mean reward: {self.mean_rewards[-1]}")
            logger.info(f"Mean discounted return: {self.mean_discounted_returns[-1]}")
            logger.info("-" * 50)

    def save_training_rewards(self, path: str) -> None:
        try:
            df = pd.read_csv(f"{path}.csv")
            df[self.seed] = self.mean_rewards
        except FileNotFoundError:
            df = pd.DataFrame(self.mean_rewards, columns=[self.seed])
            df.index.name = "iteration"
        df.to_csv(f"{path}.csv")


