
import copy
from abc import ABCMeta, abstractmethod
from dataclasses import field
from enum import StrEnum
from typing import TypeVar, Generic

import torch

from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torch import nn

from agents.base.state import State
from agents.utils import PhaseUpdatable
from agents.utils.nn.layers.linear import Linear
from agents.utils.run_configuration import RunConfiguration

Action = TypeVar('Action')
Rule = TypeVar('Rule')
Input = TypeVar('Input')


@tensorclass
class Record:
    state: State
    action: Action
    memory: TensorDict | None
    info: TensorDict = field(default_factory=lambda: TensorDict({}, batch_size=[]))


class Keys(StrEnum):
    ACTIONS = 'actions'
    VALUE = 'value'
    MEMORY = 'memory'
    ACTOR_VALUE = 'actor_value'
    POLICY = 'policy'


class Policy(Generic[Input], nn.Module, PhaseUpdatable, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        self.noise_parameters = None

    @abstractmethod
    def forward(self, state: State):
        """
        Returns: Tuple representing values and actions for state
        """
        pass

    @abstractmethod
    def select(self, state: State) -> Record:
        """
        Returns: Record with action
        """
        pass

    @abstractmethod
    def encode(self, state: State):
        """
        Returns: Tuple with hidden representation of actions and values
        """
        pass

    @abstractmethod
    def post_encode(self, state: State, output: TensorDict):
        pass

    def configure(self, configuration: RunConfiguration):
        pass

    def make_linear_layer(self, output_dim):
        return Linear(output_dim, noise_parameters=self.noise_parameters, activation='none', dropout=0, signature='')

    def clone(self):
        return copy.deepcopy(self)
