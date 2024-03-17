
from abc import ABCMeta, abstractmethod
from typing import List

from agents.utils.memory import Record

from enum import StrEnum, auto


class ValueFetchMethod(StrEnum):
    VALUE = auto()
    ACTION = auto()


class Estimator(metaclass=ABCMeta):

    def __init__(self):
        self.value_fetch_method = ValueFetchMethod.VALUE

    def update(self, value_fetch_method: ValueFetchMethod):
        self.value_fetch_method = value_fetch_method

    @property
    @abstractmethod
    def discount_factor(self) -> float:
        pass

    @abstractmethod
    def update_returns(self, records: List[Record]) -> List[Record]:
        pass

    def get_value(self, record: Record):
        match self.value_fetch_method:
            case ValueFetchMethod.VALUE:
                return record.info[Record.VALUE_KEY]
            case ValueFetchMethod.ACTION:
                return record.info[Record.ACTION_KEY][record.action]
