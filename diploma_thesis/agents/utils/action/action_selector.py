
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

import torch


class ActionSelector(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, distribution: torch.FloatTensor) -> Tuple[int, torch.FloatTensor]:
        """
        Args:
            distribution: Distribution over possible actions. Either q-values or probabilities.

        Returns: Index of the selected action and the probability of the selected action.
        """
        pass
