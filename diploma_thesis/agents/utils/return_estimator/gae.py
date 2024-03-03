
from typing import Dict
from .estimator import *


class GAE(Estimator):

    def __init__(self, discount_factor: float, lambda_: float):
        super().__init__()
        self._discount_factor = discount_factor
        self._lambda = lambda_

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    def update_returns(self, records: List[Record]) -> List[Record]:
        for i in range(len(records) - 1):
            records[i].info['advantage'] = ...
            records[i].info['return'] = ...

        return []


    @staticmethod
    def from_cli(parameters: Dict):
        return GAE(parameters['discount_factor'], parameters['lambda'])
