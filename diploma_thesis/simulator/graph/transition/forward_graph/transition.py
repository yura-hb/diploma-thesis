
from abc import ABCMeta, abstractmethod

import torch

from environment import Job


class ForwardTransition(metaclass=ABCMeta):

    @abstractmethod
    def construct(self, job: Job) -> torch.LongTensor:
        pass

