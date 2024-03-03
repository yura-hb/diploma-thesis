from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from agents.base.state import TensorState, GraphState
from .layers import Layer, from_cli as layer_from_cli


class NN(nn.Module):
    """
    A simple factory to create NN based on the list of parameters from the CLI
    """

    @dataclass
    class Configuration:

        graph: list[Layer]
        state: list[Layer]
        merge: Layer
        output: list[Layer]

        optimizer_parameters: dict

        @staticmethod
        def from_cli(parameters: dict):
            return NN.Configuration(
                graph=[layer_from_cli(layer) for layer in parameters['graph']] if parameters.get('graph') else [],
                state=[layer_from_cli(layer) for layer in parameters['state']] if parameters.get('state') else [],
                merge=layer_from_cli(parameters['merge']) if parameters.get('merge') else None,
                output=[layer_from_cli(layer) for layer in parameters['output']] if parameters.get('output') else [],
                optimizer_parameters=parameters.get('optimizer_parameters', dict())
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.state_encoder = None
        self.graph_encoder = None
        self.merge = None
        self.output = None

        self.configuration = configuration

        self.__build__()

    @property
    def is_configured(self):
        return self.model is not None

    @property
    def input_dim(self):
        return self.input_dim

    def forward(self, state):
        state = torch.atleast_2d(state)

        encoded_state = None
        encoded_graph = None

        if isinstance(state, TensorState) and self.state_encoder is not None:
            encoded_state = self.state_encoder(state.state)

        if isinstance(state, GraphState) and self.graph_encoder is not None:
            encoded_graph = self.graph_encoder(state.graph)

        hidden = self.merge(encoded_state, encoded_graph)
        output = self.output(hidden)

        return output

    def append_output_layer(self, layer: Layer):
        assert self.model is None, "Layers can be appended only before model initialization"

        self.configuration.output.append(layer)

    def parameters(self, recurse: bool = True):
        return [{'params': self.model.parameters(recurse), **self.configuration.optimizer_parameters}]

    def copy_parameters(self, other: 'NN', decay: float = 1.0):
        with torch.no_grad():
            for param, other_param in zip(self.model.parameters(), other.model.parameters()):
                param.data.copy_(param.data * (1 - decay) + other_param.data * decay)

    def clone(self):
        return deepcopy(self)

    # Utils
    def __build__(self):
        self.state_encoder = nn.Sequential(*self.configuration.state)
        self.graph_encoder = nn.Sequential(*self.configuration.graph)
        self.merge = self.configuration.merge
        self.output = nn.Sequential(*self.configuration.output)

    @staticmethod
    def from_cli(parameters: dict) -> 'NN':
        configuration = NN.Configuration.from_cli(parameters)

        return NN(configuration)
