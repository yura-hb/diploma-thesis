
from typing import Dict
from .estimator import *


class No(Estimator):

    def __init__(self, discount_factor: float):
        super().__init__()
        self._discount_factor = discount_factor

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    def update_returns(self, records: List[Record]) -> List[Record]:
        return records

    @staticmethod
    def from_cli(parameters: Dict):
        return No(parameters['discount_factor'])
