
from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch

from typing import Dict
from utils import Loggable


class ActionSelector(Loggable, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, distribution: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Args:
            distribution: Distribution over possible actions. Either q-values or probabilities.

        Returns: Index of the selected action and the probability of the selected action.
        """
        pass
