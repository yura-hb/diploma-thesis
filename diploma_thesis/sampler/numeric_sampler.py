
import torch

from .sampler import Sampler
from abc import ABCMeta, abstractmethod


class NumericSampler(Sampler, metaclass=ABCMeta):

    @property
    @abstractmethod
    def mean(self) -> torch.FloatTensor:
        pass

    @property
    @abstractmethod
    def variance(self) -> torch.FloatTensor:
        pass
