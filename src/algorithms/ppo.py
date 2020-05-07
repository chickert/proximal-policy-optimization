import logging
from copy import deepcopy
from itertools import chain
from typing import Tuple, List, Optional, Union, Iterable, Any

import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
import torch
from torch.utils.data import TensorDataset

from algorithms.param_annealing import AnnealedParam
from algorithms.timer import timer
from architectures.actor_critic import ActorCritic, DEVICE
from environment_models.base import BaseEnv

# Set up logging
logger = logging.getLogger(__name__)


class PPOLearner:

    def __init__(
            self,
            environment: BaseEnv,
            policy: ActorCritic,
            n_steps_per_trajectory: int = 16,
            n_trajectories_per_batch: int = 64,
            n_epochs: int = 10,
            n_iterations: int = 200,
            discount: float = 0.99,
            learning_rate: Union[float, AnnealedParam] = 3e-4,
            clipping_param: Union[float, AnnealedParam] = 0.2,
            critic_coefficient: Union[float, AnnealedParam] = 1.0,
            entropy_coefficient: Union[float, AnnealedParam] = 1e-2,
            bc_coefficient: Union[float, AnnealedParam] = 1e-3,
            clipping_type: str = "clamp",
            seed: int = 0
    ):
        # Set seed
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set environment and policy attributes
        self.environment = environment
        self.policy = policy

        # Set hyper-parameter attributes
        self.n_steps_per_trajectory = n_steps_per_trajectory
        self.n_trajectories_per_batch = n_trajectories_per_batch
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.discount = discount
        self.learning_rate = learning_rate
        self.clipping_param = clipping_param
        self.critic_coefficient = critic_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.bc_coefficient = bc_coefficient
        self.clipping_type = clipping_type

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(self.learning_rate))

        # Initialize attributes for performance tracking
        self.mean_rewards = []
        self.best_mean_reward = -np.inf
        self.best_policy = policy

    def calculate_discounted_returns(self, rewards: List[float]) -> List[float]:
        discounted_returns = []
        discounted_return = 0  # TODO: check if this is the correct terminal condition
        for t in reversed(range(len(rewards) - 1)):
            discounted_return = rewards[t] + self.discount*discounted_return
            discounted_returns.insert(0, discounted_return)
        return discounted_returns

    def generate_trajectory(
            self,
            use_argmax: bool = False,
            perform_reset: bool = True
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

        # Reset environment
        if perform_reset:
            self.environment.reset()

        # Calculate discounted rewards
        discounted_returns = self.calculate_discounted_returns(rewards=rewards)

        # Return states (excluding terminal state), actions, rewards and discounted rewards
        return states[:-1], actions[:-1], rewards, discounted_returns

    @timer
    def generate_batch(
            self,
            pool: Optional[mp.Pool],
            use_argmax: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:

        # Generate batch of trajectories
        if pool is None:
            trajectories = [self.generate_trajectory(use_argmax=use_argmax) for _ in range(self.n_trajectories_per_batch)]
        else:
            trajectories = pool.starmap(self.generate_trajectory, [() for _ in range(self.n_trajectories_per_batch)])

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
        states = torch.tensor(states).float().to(DEVICE).detach()
        if self.policy.actor_is_discrete:
            actions = [self.policy.inverse_action_map[tuple(action)] for action in actions]
        actions = torch.tensor(actions).float().to(DEVICE).detach()
        discounted_returns = torch.tensor(discounted_returns).float().unsqueeze(1).to(DEVICE).detach()
        old_log_probabilities = self.policy.get_distribution(states).log_prob(actions).float().to(DEVICE).detach()

        # Normalize discounted rewards
        discounted_returns = (discounted_returns - torch.mean(discounted_returns)) / (
                    torch.std(discounted_returns) + 1e-5)

        return states, actions, discounted_returns, old_log_probabilities

    @staticmethod
    def critic_loss(discounted_returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return torch.mean((discounted_returns - values).pow(2))

    def ppo_loss(
            self,
            values: torch.Tensor,
            discounted_returns: torch.Tensor,
            log_probabilities: torch.Tensor,
            old_log_probabilities: torch.Tensor,
            entropy: torch.Tensor,
            advantage_estimates: torch.Tensor
    ) -> torch.Tensor:
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        if self.clipping_type == "clamp":
            clipped_ratio = torch.clamp(ratio, 1 - self.clipping_param, 1 + self.clipping_param)
        elif self.clipping_type == "sigmoid":
            const = -logit(1/2 - self.clipping_param) / self.clipping_param
            clipped_ratio = torch.sigmoid(const * (ratio - 1)) + 0.5
        elif self.clipping_type == "tanh":
            const = np.arctanh(self.clipping_param) / self.clipping_param
            clipped_ratio = torch.tanh(const * (ratio - 1)) + 1
        elif self.clipping_type == "none":
            clipped_ratio = ratio
        else:
            raise NotImplementedError
        actor_loss = -torch.mean(
            torch.min(
                clipped_ratio * advantage_estimates,
                ratio * advantage_estimates
            )
        )
        critic_loss = self.critic_loss(discounted_returns=discounted_returns, values=values)
        entropy_loss = -torch.mean(entropy)
        return actor_loss + self.critic_coefficient*critic_loss + self.entropy_coefficient*entropy_loss

    def bc_loss(self, expert_data: TensorDataset) -> torch.Tensor:
        states, actions = map(torch.stack, zip(*expert_data))
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        expert_log_probabilities = self.policy.get_distribution(states).log_prob(actions).float().to(DEVICE)
        return -torch.mean(expert_log_probabilities)

    @timer
    def update_policy_network(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            discounted_returns: torch.Tensor,
            old_log_probabilities: torch.Tensor,
            expert_data: Optional[TensorDataset] = None,
            train_critic_only: bool = False
    ) -> None:

        for _ in range(self.n_epochs):

            # Get policy network outputs
            log_probabilities, values, entropy = self.policy(
                states=states,
                actions=actions
            )

            # Calculate loss
            if train_critic_only:
                loss = self.critic_loss(discounted_returns=discounted_returns, values=values)
            else:
                advantage_estimates = discounted_returns - values.detach()
                loss = self.ppo_loss(
                    values=values,
                    discounted_returns=discounted_returns,
                    log_probabilities=log_probabilities,
                    old_log_probabilities=old_log_probabilities,
                    entropy=entropy,
                    advantage_estimates=advantage_estimates
                )
                if expert_data:
                    loss = loss + self.bc_coefficient*self.bc_loss(expert_data=expert_data)

            # Perform gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_parameters(self) -> None:

        if type(self.learning_rate) == AnnealedParam:
            self.learning_rate = self.learning_rate.update()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = float(self.learning_rate)

        if type(self.clipping_param) == AnnealedParam:
            self.clipping_param = self.clipping_param.update()

        if type(self.critic_coefficient) == AnnealedParam:
            self.critic_coefficient = self.critic_coefficient.update()

        if type(self.entropy_coefficient) == AnnealedParam:
            self.entropy_coefficient = self.entropy_coefficient.update()

        if type(self.policy.actor_std) == AnnealedParam:
            self.policy.actor_std = self.policy.actor_std.update()

        if type(self.bc_coefficient) == AnnealedParam:
            self.bc_coefficient = self.bc_coefficient.update()

    @timer
    def train(
            self, pool: Optional[mp.Pool] = None,
            expert_data: Optional[TensorDataset] = None,
            train_critic_only_on_init: bool = False
    ) -> None:

        for i in range(self.n_iterations):

            logger.info(f"Iteration: {i + 1}")

            # Update PPO parameters
            self.update_parameters()

            # Generate batch
            states, actions, rewards, discounted_returns = self.generate_batch(pool=pool)

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
                old_log_probabilities=old_log_probabilities,
                expert_data=expert_data,
                train_critic_only=(train_critic_only_on_init and not i)
            )

            # Track and log performance
            mean_reward = np.mean(rewards)
            self.mean_rewards.append(mean_reward)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_policy = deepcopy(self.policy)
            logger.info(f"Mean reward: {'{0:.3f}'.format(mean_reward)}")
            logger.info("-" * 50)

        self.policy = self.best_policy

    def save_training_rewards(self, path: str) -> None:
        try:
            df = pd.read_csv(f"{path}.csv", index_col=0)
            df[self.seed] = self.mean_rewards
        except FileNotFoundError:
            df = pd.DataFrame(self.mean_rewards, columns=[self.seed])
            df.index.name = "iteration"
        df.to_csv(f"{path}.csv")


# Nice pattern for concatenating lists
def concatenate_lists(lists: Iterable[List[Any]]) -> List[Any]:
    return list(chain(*lists))


# Logit function
def logit(x: float) -> float:
    return np.log(x / (1 - x))


