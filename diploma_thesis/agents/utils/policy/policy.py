
import copy
from abc import ABCMeta, abstractmethod
from dataclasses import field
from typing import TypeVar, Generic

from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torch import nn

from agents.utils import PhaseUpdatable
from agents.utils.run_configuration import RunConfiguration

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
    def select(self, state, parameters):
        pass

    @property
    def is_recurrent(self):
        return False

    def configure(self, configuration: RunConfiguration):
        pass

    def clone(self):
        return copy.deepcopy(self)
