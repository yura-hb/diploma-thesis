
from typing import Dict

import torch

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
            record = records[i]
            next_record = records[i + 1] if i < len(records) - 1 else None

            next_value = 0 if next_record is None else next_record.info[Record.VALUES_KEY][records[i].action]
            value = record.info[Record.VALUES_KEY][record.action]
            advantage = record.reward + self._discount_factor * next_value - value
            next_advantage = 0 if next_record is None else next_record.info[Record.ADVANTAGE_KEY]

            records[i].info[Record.ADVANTAGE_KEY] = coef ** i * advantage + next_advantage
            records[i].info[Record.RETURN_KEY] = records[i].info[Record.ADVANTAGE_KEY] + value

        return records

    @staticmethod
    def from_cli(parameters: Dict):
        return GAE(parameters['discount_factor'], parameters['lambda'])
