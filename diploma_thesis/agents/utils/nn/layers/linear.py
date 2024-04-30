
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


class Initialization(StrEnum):
    orthogonal = auto()
    xavier = auto()


class Linear(torch.nn.modules.lazy.LazyModuleMixin, Layer):

    def __init__(self,
                 dim: int,
                 activation: str,
                 dropout: float,
                 noise_parameters: Dict,
                 signature: str,
                 initialization: Initialization = Initialization.orthogonal.value):
        super().__init__(signature=signature)

        self.dim = dim
        self.activation = Activation(kind=activation, signature='')
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.initialization = initialization
        self.noise_parameters = noise_parameters or dict()

        self.__build__()

    def initialize_parameters(self, input) -> None:
        self.linear.initialize_parameters(input)

        if isinstance(self.linear, torch.nn.Linear):
            match self.initialization:
                case Initialization.orthogonal.value:
                    torch.nn.init.orthogonal_(self.linear.weight)
                    torch.nn.init.zeros_(self.linear.bias)
                case _:
                    pass

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        batch = self.linear(batch)
        batch = self.activation(batch)

        if self.dropout is not None:
            batch = self.dropout(batch)

        return batch

    def to_noisy(self, noise_parameters):
        self.noise_parameters = noise_parameters

        self.__build__()

    def __build__(self):
        kind = self.noise_parameters.get('kind', NoiseType.none)

        if kind != NoiseType.none:
            kind = NoiseType(kind)

        parameters = self.noise_parameters.get('parameters', dict())

        match kind:
            case NoiseType.none:
                self.linear = nn.LazyLinear(out_features=self.dim)
            case NoiseType.independent:
                self.linear = IndependentNoisyLayer(output_features=self.dim, **parameters)
            case NoiseType.factorized:
                self.linear = FactorisedNoisyLayer(output_features=self.dim, **parameters)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Linear(
            dim=parameters['dim'],
            activation=parameters['activation'],
            dropout=parameters.get('dropout'),
            noise_parameters=parameters.get('noise', dict()).get('parameters', dict()),
            signature=parameters['signature'],
            initialization=parameters.get('initialization')
        )
