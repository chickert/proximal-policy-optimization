from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):

    def __init__(
            self,
            state_space_dimension: int,
            action_map: Dict[int, np.array],
            actor_hidden_layer_units: List[int],
            critic_hidden_layer_units: List[int],
            non_linearity: nn.Module = nn.ReLU
    ):
        super(ActorCritic, self).__init__()

        # Make actor network
        action_space_dimension = len(action_map)
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

    def get_argmax_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state).float()
        action = self.actor(state).argmax().item()
        return self.action_map[action]
