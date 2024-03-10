
import torch
from .layer import *
from .activation import Activation
from .noisy import IndependentNoisyLayer, FactorisedNoisyLayer
from enum import StrEnum, auto
from typing import Dict


class NoiseType(StrEnum):
    none = auto()
    independent = auto()
    factorized = auto()


class Linear(Layer):

    def __init__(self,
                 dim: int,
                 activation: str,
                 dropout: float,
                 noise: NoiseType = NoiseType.none,
                 noise_parameters: Dict = None):
        super().__init__()

        self.dim = dim
        self.activation = Activation(kind=activation)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        if noise_parameters is None:
            noise_parameters = dict()

        match noise:
            case NoiseType.none:
                self.linear = nn.LazyLinear(out_features=dim)
            case NoiseType.independent:
                self.linear = IndependentNoisyLayer(output_features=dim, **noise_parameters)
            case NoiseType.factorized:
                self.linear = FactorisedNoisyLayer(output_features=dim, **noise_parameters)

    def __call__(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        batch = self.linear(batch)
        batch = self.activation(batch)

        if self.dropout is not None:
            batch = self.dropout(batch)

        return batch

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Linear(
            dim=parameters['dim'],
            activation=parameters['activation'],
            dropout=parameters.get('dropout'),
            noise=parameters.get('noise', dict()).get('kind', NoiseType.none),
            noise_parameters=parameters.get('noise', dict()).get('parameters', dict())
        )
