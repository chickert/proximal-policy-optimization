
from typing import List

import torch
import torch.nn as nn

# Set up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MultiLayerPerception(nn.Sequential):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_layer_units: List[int],
            activation: nn.Module,
            softmax_output: bool = False
    ):
        layers = [
            nn.Linear(in_features, hidden_layer_units[0]).to(DEVICE),
            activation().to(DEVICE)
        ]

        for i in range(1, len(hidden_layer_units)):
            layers += [
                nn.Linear(hidden_layer_units[i - 1], hidden_layer_units[i]).to(DEVICE),
                activation().to(DEVICE),
            ]
        layers += [
            nn.Linear(hidden_layer_units[-1], out_features).to(DEVICE)
        ]
        if softmax_output:
            layers += [nn.Softmax(dim=-1).to(DEVICE)]
        super().__init__(*layers)

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path) -> None:
        self.load_state_dict(torch.load(path))

