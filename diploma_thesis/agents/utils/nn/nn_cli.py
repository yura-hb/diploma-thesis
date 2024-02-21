
import torch

from torch import nn
from dataclasses import dataclass
from copy import deepcopy


class NNCLI(nn.Module):
    """
    A simple factory to create NN based on the list of parameters from the CLI
    """

    @dataclass
    class Configuration:

        @dataclass
        class Layer:
            pass

        @dataclass
        class InstanceNorm:

            @staticmethod
            def from_cli(parameters: dict):
                return NNCLI.Configuration.InstanceNorm()

        @dataclass
        class Linear:
            dim: int
            activation: str = 'none'
            dropout: float = 0.0

            @staticmethod
            def from_cli(parameters: dict):
                return NNCLI.Configuration.Linear(
                    dim=parameters['dim'],
                    activation=parameters['activation'],
                    dropout=parameters.get('dropout', 0.0)
                )

        layers: list[Layer]

        optimizer_parameters: dict

        @staticmethod
        def from_cli(parameters: dict):
            key_to_cls = {
                'linear': NNCLI.Configuration.Linear,
                'instance_norm': NNCLI.Configuration.InstanceNorm
            }

            return NNCLI.Configuration(
                layers=[
                    key_to_cls[layer['kind']].from_cli(layer.get('parameters', dict()))
                    for layer in parameters['layers']
                ],
                optimizer_parameters=parameters.get('optimizer_parameters', dict())
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.model = None
        self._input_dim = None
        self.configuration = configuration

    @property
    def is_connected(self):
        return self.model is not None

    @property
    def input_dim(self):
        return self.input_dim

    def connect(self, input_dim: int, output_layer: Configuration.Linear):
        self.model = nn.Sequential()
        self._input_dim = input_dim

        previous_dim = input_dim

        for layer in self.configuration.layers + [output_layer]:
            layer, output_dim = self.__make_layer__(previous_dim, layer)

            self.model = self.model.append(layer)

            previous_dim = output_dim

    def forward(self, x):
        return self.model(torch.atleast_2d(x))

    def parameters(self, recurse: bool = True):
        return [{'params': self.model.parameters(recurse), **self.configuration.optimizer_parameters}]

    def copy_parameters(self, other: 'NNCLI', decay: float = 1.0):
        with torch.no_grad():
            for param, other_param in zip(self.model.parameters(), other.model.parameters()):
                param.data.copy_(param.data * (1 - decay) + other_param.data * decay)

    def clone(self):
        return deepcopy(self)

    # Utils

    def __make_layer__(self, input_dim, layer: Configuration.Layer):
        match layer:
            case NNCLI.Configuration.InstanceNorm():
                return nn.InstanceNorm1d(input_dim), input_dim
            case NNCLI.Configuration.Linear(output_dim, activation, dropout):
                return self.__make_linear_layer__(input_dim, output_dim, activation, dropout), output_dim
            case _:
                raise ValueError(f"Unknown layer type {layer}")

    def __make_linear_layer__(self, input_dim, output_dim, activation, dropout):
        result = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

        if activation := self.__make_activation__(activation):
            result = result.append(activation)

        if dropout > 0:
            result = result.append(nn.Dropout(dropout))

        return result

    @staticmethod
    def __make_activation__(activation: str):
        match activation:
            case 'relu':
                return nn.ReLU()
            case 'tanh':
                return nn.Tanh()
            case 'sigmoid':
                return nn.Sigmoid()
            case 'softmax':
                return nn.Softmax(dim=1)
            case 'none':
                return None
            case _:
                raise ValueError(f"Unknown activation function {activation}")

    @staticmethod
    def from_cli(parameters: dict) -> nn.Module:
        configuration = NNCLI.Configuration.from_cli(parameters)

        return NNCLI(configuration)

