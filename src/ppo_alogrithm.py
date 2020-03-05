from typing import Tuple, Union, Dict, List

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box
from torch.distributions import Categorical

from robot_environments.pusher import PusherEnv
from robot_environments.reacher import ReacherEnv
from robot_environments.reacher_wall import ReacherWallEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOL = 1e-10

class ActorCritic(nn.Module):

    def __init__(
            self,
            state_space_dimension: int,
            action_space_dimension: int,
            action_map: Dict[int, np.array],
            actor_hidden_layer_units: List[int],
            critic_hidden_layer_units: List[int],
            non_linearity: nn.Module = nn.ReLU
    ):
        super(ActorCritic, self).__init__()

        # Make actor network
        actor_layers = [
            nn.Linear(state_space_dimension, actor_hidden_layer_units[0]),
            non_linearity()
        ]
        for i in range(1, len(actor_hidden_layer_units)):
            actor_layers += [
                nn.Linear(actor_hidden_layer_units[i - 1], actor_hidden_layer_units[i]),
                non_linearity()
            ]
        actor_layers += [
            nn.Linear(actor_hidden_layer_units[-1], action_space_dimension),
            nn.Softmax(dim=-1)
        ]
        self.actor = nn.Sequential(*actor_layers)

        # Make critic network
        critic_layers = [
            nn.Linear(state_space_dimension, critic_hidden_layer_units[0]),
            non_linearity()
        ]
        for i in range(1, len(critic_hidden_layer_units)):
            critic_layers += [
                nn.Linear(critic_hidden_layer_units[i - 1], critic_hidden_layer_units[i]),
                non_linearity()
            ]
        critic_layers += [
            nn.Linear(critic_hidden_layer_units[-1], 1)
        ]
        self.critic = nn.Sequential(*critic_layers)

        # Initialize other attributes
        self.action_map = action_map

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_probabilities = self.actor(states)
        values = self.critic(states)
        return policy_probabilities, values

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state).float()
        action = Categorical(self.actor(state)).sample().item()
        return self.action_map[action]

    def get_value_estimates_as_array(self, states: int = List[np.ndarray]) -> np.array:
        states = torch.tensor(states).float()
        return self.actor(states).detach().numpy()


class PPOLearner:

    def __init__(
            self,
            environment: Union[ReacherEnv, ReacherWallEnv, PusherEnv],
            state_space_dimension: int,
            action_space_dimension: int,
            action_map: Dict[int, np.array],
            critic_hidden_layer_units: List[int],
            actor_hidden_layer_units: List[int],
            random_init_box: Box,
            n_steps_per_trajectory: int = 256,
            n_trajectories_per_batch: int = 16,
            n_epochs: int = 5,
            n_iterations: int = 100,
            learning_rate: float = 2e-3,
            discount: float = 0.99,
            gae_param: float = 0.95,
            clipping_param: float = 0.2,
            critic_coefficient: float = 0.5,
            entropy_coefficient: float = 1e-3,
            randomize_init_state: bool = True
    ):
        # Set environment attribute
        self.environment = environment

        # Initialize actor-critic policy network
        self.policy = ActorCritic(
            state_space_dimension=state_space_dimension,
            action_space_dimension=action_space_dimension,
            action_map=action_map,
            actor_hidden_layer_units=actor_hidden_layer_units,
            critic_hidden_layer_units=critic_hidden_layer_units,
        )

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Set hyperparameter attributes
        self.random_init_box = random_init_box
        self.n_steps_per_trajectory = n_steps_per_trajectory
        self.n_trajectories_per_batch = n_trajectories_per_batch
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.discount = discount
        self.gae_param = gae_param
        self.clipping_param = clipping_param
        self.critic_coefficient = critic_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.randomize_init_state = randomize_init_state

    def generate_trajectory(self) -> Tuple[List[np.ndarray], List[float]]:

        # Initialize trajectory
        states = []
        rewards = []

        # Set initial state
        init_state = self.environment.init
        if self.randomize_init_state:
            state = self.random_init_box.sample()
        else:
            state = init_state
        self.environment.init = state
        self.environment.reset()

        # Generate a sample trajectory under the current policy
        for _ in range(self.n_steps_per_trajectory + 1):

            # Sample from policy and receive feedback from environment
            action = self.policy.sample_action(state)
            new_state, reward, done, info = self.environment.step(action)

            # Store information from step
            states.append(state)
            rewards.append(reward)

            # Update state
            state = new_state

        # Reset environment initial state
        self.environment.init = init_state
        self.environment.reset()

        return states, rewards

    def calculate_advantage_estimates(self, states: List[np.array], rewards: List[float]) -> List[float]:
        values = self.policy.get_value_estimates_as_array(states)
        advantage_estimates = []
        advantage = 0
        for t in reversed(range(self.n_steps_per_trajectory)):
            delta = rewards[t] + self.discount*values[t + 1] - values[t]
            advantage = delta + self.discount*self.gae_param*advantage
            advantage_estimates.insert(0, advantage)
        return advantage_estimates

    def calculate_discounted_rewards(self, states: List[np.array], rewards: List[float]) -> List[float]:
        discounted_rewards = []
        discounted_reward = 0
        for t in reversed(range(self.n_steps_per_trajectory)):
            discounted_reward = rewards[t] + self.discount*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        return discounted_rewards

    def generate_batch(self) -> Tuple[List[np.ndarray], List[float], List[float]]:

        states = []
        rewards = []
        discounted_rewards = []

        for _ in range(self.n_trajectories_per_batch):

            # Generate trajectory under current policy
            new_states, new_rewards = self.generate_trajectory()

            # Add advantages and policy probabilities to batch
            states += new_states[:-1]
            rewards += new_rewards
            discounted_rewards += self.calculate_discounted_rewards(
                states=new_states,
                rewards=new_rewards
            )

        return states, rewards, discounted_rewards

    def train(self, log: bool = True) -> None:

        for i in range(self.n_iterations):

            # Generate batch
            states, rewards, discounted_rewards = self.generate_batch()

            # Convert data to tensors
            states = torch.tensor(states).float().to(DEVICE).detach()
            discounted_rewards = torch.tensor(discounted_rewards).float().unsqueeze(1).to(DEVICE).detach()
            old_policy_probabilities = self.policy.actor(states).float().to(DEVICE).detach()

            # Normalize rewards
            discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) \
                                  / (torch.std(discounted_rewards) + TOL)

            for _ in range(self.n_epochs):

                # Get policy network outputs
                policy_probabilities, values = self.policy(states)

                # Get advantage estimates
                advantage_estimates = (discounted_rewards - values.detach())

                # Compute loss
                ratio = torch.exp(torch.log(policy_probabilities + TOL) - torch.log(old_policy_probabilities + TOL))
                actor_loss = -torch.mean(
                    torch.min(
                        torch.clamp(ratio, 1 - self.clipping_param, 1 + self.clipping_param) * advantage_estimates,
                        ratio * advantage_estimates
                    )
                )
                critic_loss = torch.mean((discounted_rewards - values).pow(2))
                entropy_loss = torch.mean(policy_probabilities * torch.log(policy_probabilities))
                loss = (actor_loss + self.critic_coefficient*critic_loss + self.entropy_coefficient*entropy_loss)

                # Perform gradient update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if log:
                print(f"Iteration: {i + 1}")
                print(f"Mean reward: {np.mean(rewards)}")
                print("-"*50)






