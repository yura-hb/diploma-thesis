
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

        # Drop the last record to preserve the value of the last state
        for i in reversed(range(len(records) - 1)):
            record = records[i]
            next_record = records[i + 1]

            next_value = 0 if next_record is None else self.get_value(next_record)
            value = self.get_value(record)
            advantage = record.reward + self._discount_factor * next_value - value

            if Record.ADVANTAGE_KEY in next_record.info.keys():
                next_advantage = next_record.info[Record.ADVANTAGE_KEY]
            else:
                next_advantage = next_record.info[Record.ACTION_KEY][next_record.action] - self.get_value(next_record)

            records[i].info[Record.ADVANTAGE_KEY] = coef ** i * advantage + next_advantage
            records[i].info[Record.RETURN_KEY] = records[i].info[Record.ADVANTAGE_KEY] + value



        return records[:-1]

    @staticmethod
    def from_cli(parameters: Dict):
        return GAE(parameters['discount_factor'], parameters['lambda'])
