
from abc import ABCMeta, abstractmethod
from typing import List

from agents.utils.memory import Record


class Estimator(metaclass=ABCMeta):

    @property
    @abstractmethod
    def discount_factor(self) -> float:
        pass

    @abstractmethod
    def update_returns(self, records: List[Record]) -> List[Record]:
        pass
