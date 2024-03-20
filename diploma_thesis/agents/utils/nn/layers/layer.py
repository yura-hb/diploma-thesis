
from abc import ABCMeta

from torch import nn


class Layer(nn.Module, metaclass=ABCMeta):

    def __init__(self, signature: str):
        super().__init__()

        self.signature_ = signature

    @property
    def signature(self):
        return self.signature_

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        pass
