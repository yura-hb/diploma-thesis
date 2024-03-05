from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from agents.base.state import TensorState, GraphState
from .layers import Layer, from_cli as layer_from_cli, Merge


class NeuralNetwork(nn.Module):
    """
    A simple factory to create NN based on the list of parameters from the CLI
    """

    @dataclass
    class Configuration:

        graph: list[Layer]
        state: list[Layer]
        merge: Layer
        output: list[Layer]

        @staticmethod
        def from_cli(parameters: dict):
            return NeuralNetwork.Configuration(
                graph=[layer_from_cli(layer) for layer in parameters['graph']] if parameters.get('graph') else [],
                state=[layer_from_cli(layer) for layer in parameters['state']] if parameters.get('state') else [],
                merge=layer_from_cli(parameters['merge']) if parameters.get('merge') else Merge(),
                output=[layer_from_cli(layer) for layer in parameters['output']] if parameters.get('output') else []
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.state_encoder = None
        self.graph_encoder = None
        self.merge = None
        self.output = None

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
        encoded_state = None
        encoded_graph = None

        if isinstance(state, TensorState) and self.state_encoder is not None:
            encoded_state = self.state_encoder(torch.atleast_2d(state.state))

        if isinstance(state, GraphState) and self.graph_encoder is not None:
            data = state.graph.data

            if not self.is_configured:
                if isinstance(data, HeteroData):
                    self.graph_encoder = to_hetero(self.graph_encoder, data.metadata(), aggr='sum')

            encoded_graph = self.graph_encoder(data.x_dict, data.edge_index_dict)

        hidden = self.merge(encoded_state, encoded_graph)
        output = self.output(hidden)

        _is_configured = True

        return output

    def append_output_layer(self, layer: Layer):
        assert not self.is_configured, "The model is already configured"

        self.output.append(layer)

    def copy_parameters(self, other: 'NeuralNetwork', decay: float = 1.0):
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
    def from_cli(parameters: dict) -> 'NeuralNetwork':
        configuration = NeuralNetwork.Configuration.from_cli(parameters)

        return NeuralNetwork(configuration)
