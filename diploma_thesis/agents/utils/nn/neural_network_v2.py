
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch_geometric as pyg
from torch import nn

from tensordict import TensorDict

from .layers import Layer, from_cli as layer_from_cli, Merge
from .layers.linear import Linear


class NeuralNetwork(nn.Module):

    """
    A simple factory to create NN based on the list of parameters from the CLI
    """

    @dataclass
    class Configuration:
        signature: str
        layers: List[Tuple[Layer, str] | Layer]

        @staticmethod
        def from_cli(parameters: dict):
            return NeuralNetwork.Configuration(
                ...
                # graph=[layer_from_cli(layer) for layer in parameters['graph']] if parameters.get('graph') else [],
                # state=[layer_from_cli(layer) for layer in parameters['state']] if parameters.get('state') else [],
                # merge=layer_from_cli(parameters['merge']) if parameters.get('merge') else Merge(),
                # output=[layer_from_cli(layer) for layer in parameters['output']] if parameters.get('output') else []
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        model = None

        # self.state_encoder = None
        # self.graph_encoder = None
        # self.merge = None
        # self.output = None

        self.configuration = configuration
        self._is_configured = False

        self.__build__()

    @property
    def is_configured(self):
        return self._is_configured

    @property
    def input_dim(self):
        return self.input_dim

    def forward(self, state):
        output = self.__forward__(state)

        _is_configured = True

        return output

    def to_noisy(self, noise_parameters):
        assert not self.is_configured, "The model is already configured"

        for layer in self.configuration.layers:
            if (isinstance(layer, Tuple) and isinstance(layer[0], Linear)) or isinstance(layer, Linear):
                layer.to_noisy(noise_parameters)

        self.__build__()

    def copy_parameters(self, other: 'NeuralNetwork', decay: float = 1.0):
        with torch.no_grad():
            for param, other_param in zip(self.model.parameters(), other.model.parameters()):
                param.data.copy_(param.data * (1 - decay) + other_param.data * decay)

    def clone(self):
        return deepcopy(self)

    # Utils
    def __build__(self):
        self.model = pyg.nn.Sequential(
            'state, graph, memory, output, new_memory -> output, new_memory',
            self.configuration.layers
        )

    def __forward__(self, state):
        memory, output = self.model(
            state.state, state.graph, state.memory, TensorDict({}, batch_size=state.batch_size)
        )

        return memory, output

    @staticmethod
    def from_cli(parameters: dict) -> 'NeuralNetwork':
        configuration = NeuralNetwork.Configuration.from_cli(parameters)

        return NeuralNetwork(configuration)
