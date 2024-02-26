
import torch

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from tensordict.prototype import tensorclass
from typing import TypeVar, Generic
from utils import Loggable

State = TypeVar('State')
Input = TypeVar('Input')
Action = TypeVar('Action')
Result = TypeVar('Result')


class Model(Loggable, Generic[Input, State, Action, Result], metaclass=ABCMeta):

    @tensorclass
    class Record:
        result: Result
        state: State
        action: Action
        action_values: torch.FloatTensor

    @abstractmethod
    def __call__(self, state: State, parameters: Input) -> Record:
        pass


class NNModel(Model[Input, State, Action, Result], metaclass=ABCMeta):

    @abstractmethod
    def predict(self, state: State) -> torch.FloatTensor:
        pass

    @abstractmethod
    def parameters(self, recurse: bool = True):
        pass

    @abstractmethod
    def copy_parameters(self, other, decay: float = 1.0):
        pass

    @abstractmethod
    def clone(self):
        pass
