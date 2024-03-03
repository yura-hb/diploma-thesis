
import torch
from .layer import *
from .activation import Activation


class Linear(Layer):

    def __init__(self, dim: int, activation: str, dropout: float = None):
        super().__init__()

        self.dim = dim
        self.activation = Activation(kind=activation)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.linear = nn.LazyLinear(out_features=dim)

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
            dropout=parameters.get('dropout')
        )
