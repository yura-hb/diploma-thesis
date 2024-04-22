from .layer import *
import torch


class Activation(Layer):

    def __init__(self, kind: str, signature: str):
        super().__init__(signature=signature)

        self.kind = kind
        self.activation = self.__make_activation__()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.activation is None:
            return x

        return self.activation(x)

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
            case 'leaky_relu':
                return nn.LeakyReLU()
            case 'none':
                return None
            case _:
                raise ValueError(f"Unknown activation function {self.kind}")

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Activation(kind=parameters['kind'], signature=parameters['signature'])
