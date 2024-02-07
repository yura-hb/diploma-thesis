from abc import abstractmethod

from typing import TypeVar, Generic

Input = TypeVar('Input')
State = TypeVar('State')


class Encoder(Generic[Input, State]):

    @abstractmethod
    def encode(self, parameters: Input) -> State:
        pass
