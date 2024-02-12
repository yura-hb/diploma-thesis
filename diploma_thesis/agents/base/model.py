
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic
from agents.utils import Loggable

State = TypeVar('State')
Input = TypeVar('Input')
Action = TypeVar('Action')
Result = TypeVar('Result')


class Model(Loggable, Generic[Input, State, Action, Result], metaclass=ABCMeta):

    @dataclass
    class Record:
        result: Result
        state: State
        action: Action

    @abstractmethod
    def __call__(self, state: State, parameters: Input) -> Record:
        pass
