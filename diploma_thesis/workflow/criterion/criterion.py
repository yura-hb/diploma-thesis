
from abc import ABCMeta, abstractmethod
from enum import StrEnum, auto


class Direction(StrEnum):
    minimize = auto()
    maximize = auto()


class Scale(StrEnum):
    linear = auto()
    log = auto()
    exp = auto()


class Criterion(metaclass=ABCMeta):

    def __init__(self, weight: float, direction: Direction, scale: Scale):
        self._weight = weight
        self._direction = direction
        self._scale = scale

    @classmethod
    @property
    @abstractmethod
    def key(cls) -> str:
        pass

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def scale(self) -> Scale:
        return self._scale

    @abstractmethod
    def compute(self, statistics):
        pass

    @staticmethod
    def parameters_from_cli(parameters: dict):
        return {
            'weight': parameters['weight'],
            'direction': Direction[parameters['direction']],
            'scale': Scale[parameters['scale']]
        }

    @classmethod
    def from_cli(cls, parameters: dict):
        return cls(**Criterion.parameters_from_cli(parameters))
