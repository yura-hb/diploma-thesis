
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch_geometric as pyg
from torch import nn

from .layers import Key
from .layers import Layer, from_cli as layer_from_cli
from .layers.linear import Linear


class NeuralNetwork(nn.Module):

    @dataclass
    class Configuration:
        layers: List[Layer]
        init_seed: int

        @staticmethod
        def from_cli(parameters: dict):
            return NeuralNetwork.Configuration(
                layers=[layer_from_cli(layer) for layer in parameters['layers']] if parameters.get('layers') else [],
                init_seed=parameters.get('init_seed', 0)
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.is_configured = False

        self.configuration = configuration

        self.__build__()

    def forward(self, state):
        if not self.is_configured:
            torch.manual_seed(self.configuration.init_seed)

        output = self.__forward__(state)

        self.is_configured = True

        return output

    def to_noisy(self, noise_parameters):
        assert not self.is_configured, "The model is already configured"

        for layer in self.configuration.layers:
            if (isinstance(layer, Tuple) and isinstance(layer[0], Linear)) or isinstance(layer, Linear):
                layer.to_noisy(noise_parameters)

        self.__build__()

    def clone(self):
        return deepcopy(self)

    # Utils
    def __build__(self):
        signature = f'{Key.STATE}, {Key.GRAPH}, {Key.MEMORY}'

        self.model = pyg.nn.Sequential(
            signature,
            [(layer, layer.signature) for layer in self.configuration.layers]
        )

    def __forward__(self, state):
        output = self.model(state.state, state.graph, state.memory)

        return output

    @staticmethod
    def from_cli(parameters: dict) -> 'NeuralNetwork':
        configuration = NeuralNetwork.Configuration.from_cli(parameters)

        return NeuralNetwork(configuration)
