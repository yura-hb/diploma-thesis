
from abc import ABCMeta, abstractmethod
from dataclasses import field
from typing import TypeVar, Generic, Tuple

import torch
from torch import nn
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
    info: TensorDict = field(default_factory=lambda: TensorDict({}, batch_size=[]))


class Policy(Generic[Input], nn.Module, PhaseUpdatable, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, state: State, parameters: Input) -> Record:
        pass

    @abstractmethod
    def predict(self, state: State) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    @abstractmethod
    def clone(self):
        pass
