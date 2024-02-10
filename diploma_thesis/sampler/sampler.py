

from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch


class Sampler(metaclass=ABCMeta):

    @abstractmethod
    def sample(self, shape: torch.Size | Tuple) -> torch.FloatTensor:
        pass
