

from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch


class Sampler(metaclass=ABCMeta):

    def __init__(self):
        self.generator = torch.default_generator

    def connect(self, generator):
        self.generator = generator

    @abstractmethod
    def sample(self, shape: torch.Size | Tuple) -> torch.FloatTensor:
        pass
