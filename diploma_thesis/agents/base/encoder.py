from abc import abstractmethod
from typing import TypeVar, Generic

from utils import Loggable

Input = TypeVar('Input')
State = TypeVar('State')


class Encoder(Loggable, Generic[Input, State]):

    @abstractmethod
    def encode(self, parameters: Input) -> State:
        pass
