from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class ActorCritic(nn.Module):

    def __init__(
            self,
            state_space_dimension: int,
            action_space_dimension: int,
            actor_hidden_layer_units: List[int],
            critic_hidden_layer_units: List[int],
            discrete_actor: bool,
            action_map: Optional[Dict[int, np.ndarray]],
            actor_std: float,
            non_linearity: nn.Module = nn.ReLU,
    ):
        super(ActorCritic, self).__init__()

        # Define policy as discrete or continuous
        self.discrete_actor = discrete_actor
        if self.discrete_actor:
            self.action_map = action_map
            self.inverse_action_map = {tuple(action): key for key, action in action_map.items()}
        else:
            self.actor_std = actor_std

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
        ]
        if self.discrete_actor:
            actor_layers += [nn.Softmax(dim=-1)]
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

    def get_distribution(self, states: torch.Tensor):
        if self.discrete_actor:
            return Categorical(self.actor(states))
        else:
            return Normal(loc=self.actor(states), scale=self.actor_std)

    def get_distribution_argmax(self, states: torch.Tensor):
        if self.discrete_actor:
            return self.actor(states).argmax()
        else:
            return self.actor(states)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(states)
        log_probabilities = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states)
        return log_probabilities, values, entropy

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state).float()
        action = self.get_distribution(state).sample()
        if self.discrete_actor:
            return self.action_map[action.item()]
        else:
            return action.detach().numpy()

    def get_argmax_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state).float()
        action = self.get_distribution_argmax(state)
        if self.discrete_actor:
            return self.action_map[action.item()]
        else:
            return action.detach().numpy()
