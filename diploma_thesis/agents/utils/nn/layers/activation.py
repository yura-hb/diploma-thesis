from .layer import *
import torch


class Activation(Layer):

    def __init__(self, kind: str):
        super().__init__()

        self.kind = kind
        self.activation = self.__make_activation__()

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        if self.activation is None:
            return batch

        return self.activation(batch)

    def __make_activation__(self):
        match self.kind:
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
                raise ValueError(f"Unknown activation function {self.kind}")

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Activation(kind=parameters['kind'])
