from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from algorithm.annealing import AnnealedParam

# Set up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic(nn.Module):

    def __init__(
            self,
            state_space_dimension: int,
            action_space_dimension: int,
            actor_hidden_layer_units: List[int],
            critic_hidden_layer_units: List[int],
            action_map: Optional[Dict[int, np.ndarray]],
            actor_std: Union[float, AnnealedParam],
            non_linearity: nn.Module = nn.ReLU,
    ):
        super(ActorCritic, self).__init__()

        # Define policy as discrete or continuous
        self.action_map = action_map
        if self.action_map:
            self.actor_is_discrete = True
            self.inverse_action_map = {tuple(action): key for key, action in action_map.items()}
        else:
            self.actor_is_discrete = False
        self.actor_std = actor_std

        # Make actor network
        actor_layers = [
            nn.Linear(state_space_dimension, actor_hidden_layer_units[0]).to(DEVICE),
            non_linearity().to(DEVICE)
        ]
        for i in range(1, len(actor_hidden_layer_units)):
            actor_layers += [
                nn.Linear(actor_hidden_layer_units[i - 1], actor_hidden_layer_units[i]).to(DEVICE),
                non_linearity().to(DEVICE)
            ]
        actor_layers += [
            nn.Linear(actor_hidden_layer_units[-1], action_space_dimension).to(DEVICE),
        ]
        if self.actor_is_discrete:
            actor_layers += [nn.Softmax(dim=-1).to(DEVICE)]
        self.actor = nn.Sequential(*actor_layers)

        # Make critic network
        critic_layers = [
            nn.Linear(state_space_dimension, critic_hidden_layer_units[0]).to(DEVICE),
            non_linearity().to(DEVICE)
        ]
        for i in range(1, len(critic_hidden_layer_units)):
            critic_layers += [
                nn.Linear(critic_hidden_layer_units[i - 1], critic_hidden_layer_units[i]).to(DEVICE),
                non_linearity().to(DEVICE)
            ]
        critic_layers += [
            nn.Linear(critic_hidden_layer_units[-1], 1).to(DEVICE)
        ]
        self.critic = nn.Sequential(*critic_layers).to(DEVICE)

    def get_distribution(self, states: torch.Tensor):
        if self.actor_is_discrete:
            return Categorical(self.actor(states).to(DEVICE))
        else:
            return Normal(loc=self.actor(states).to(DEVICE), scale=self.actor_std)

    def get_distribution_argmax(self, states: torch.Tensor):
        if self.actor_is_discrete:
            return self.actor(states).to(DEVICE).argmax()
        else:
            return self.actor(states).to(DEVICE)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(states)
        log_probabilities = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).to(DEVICE)
        return log_probabilities, values, entropy

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state).float().to(DEVICE)
        action = self.get_distribution(state).sample()
        if DEVICE == "cuda":
            action = action.cpu()
        if self.actor_is_discrete:
            return self.action_map[action.item()]
        else:
            return action.detach().numpy()

    def get_argmax_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state).float().to(DEVICE)
        action = self.get_distribution_argmax(state)
        if DEVICE == "cuda":
            action = action.cpu()
        if self.actor_is_discrete:
            return self.action_map[action.item()]
        else:
            return action.detach().numpy()
