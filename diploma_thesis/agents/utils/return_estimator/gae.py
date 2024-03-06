
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
        coef = self._discount_factor * self._lambda

        for i in reversed(range(len(records))):
            next_value = 0 if i == len(records) - 1 else records[i + 1].info[Record.ADVANTAGE_KEY]
            value = records[i].info[Record.VALUES_KEY]
            advantage = records[i].reward + self._discount_factor * next_value - value
            next_advantage = 0 if i == len(records) - 1 else records[i + 1].info[Record.ADVANTAGE_KEY]

            records[i].info[Record.ADVANTAGE_KEY] = coef ** i * advantage + next_advantage
            records[i].info[Record.RETURN_KEY] = records[i].info[Record.ADVANTAGE_KEY] + value

        return records

    @staticmethod
    def from_cli(parameters: Dict):
        return GAE(parameters['discount_factor'], parameters['lambda'])
