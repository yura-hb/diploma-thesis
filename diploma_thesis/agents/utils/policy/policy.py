
from abc import ABCMeta, abstractmethod
from dataclasses import field
from typing import TypeVar, Generic

import torch
from tensordict import TensorDict
from tensordict.prototype import tensorclass

from agents.utils import PhaseUpdatable

State = TypeVar('State')
Action = TypeVar('Action')
Rule = TypeVar('Rule')
Input = TypeVar('Input')


@tensorclass
class Record:
    state: State
    action: Action
    info: TensorDict = field(default_factory=lambda: TensorDict(batch_size=[]))


class Policy(Generic[Input], PhaseUpdatable, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, state: State, parameters: Input) -> Record:
        pass

    @abstractmethod
    def predict(self, state: State) -> torch.FloatTensor:
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def parameters(self, recurse: bool = True):
        pass

    @abstractmethod
    def copy_parameters(self, other: 'Policy', decay: float = 1.0):
        pass
