from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from architectures.mlp import MultiLayerPerception, DEVICE
from algorithms.param_annealing import AnnealedParam


class ActorCritic(nn.Module):

    def __init__(
            self,
            state_space_dimension: int,
            action_space_dimension: int,
            actor_hidden_layer_units: List[int],
            critic_hidden_layer_units: List[int],
            action_map: Optional[Dict[int, np.ndarray]],
            actor_std: Union[float, AnnealedParam],
            activation: nn.Module
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
        self.actor = MultiLayerPerception(
            in_features=state_space_dimension,
            out_features=action_space_dimension,
            hidden_layer_units=actor_hidden_layer_units,
            activation=activation,
            softmax_output=self.actor_is_discrete
        ).to(DEVICE)

        # Make critic network
        self.critic = MultiLayerPerception(
            in_features=state_space_dimension,
            out_features=1,
            hidden_layer_units=critic_hidden_layer_units,
            activation=activation,
            softmax_output=False
        ).to(DEVICE)

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
