
from abc import ABCMeta, abstractmethod

from torch import nn


class Layer(nn.Module, metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        pass
