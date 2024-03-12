
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic

import torch
from tensordict.prototype import tensorclass

from agents.utils.policy import Policy, PolicyRecord
from agents.utils import Phase, PhaseUpdatable
from utils import Loggable
from dataclasses import dataclass

State = TypeVar('State')
Input = TypeVar('Input')
Rule = TypeVar('Rule')
Action = TypeVar('Action')
Result = TypeVar('Result')


class Model(Loggable, Generic[Input, State, Action, Result], metaclass=ABCMeta):

    @tensorclass
    class Record:
        result: Result
        record: PolicyRecord | None

    @abstractmethod
    def __call__(self, state: State, parameters: Input) -> Record:
        pass


class DeepPolicyModel(Model[Input, State, Action, Result], PhaseUpdatable, metaclass=ABCMeta):

    @dataclass
    class Configuration:
        compile: bool = False

        @staticmethod
        def from_cli(parameters):
            return DeepPolicyModel.Configuration(compile=parameters.get('compile', True))

    def __init__(self, policy: Policy[Input], configuration: Configuration):
        super().__init__()

        self.policy = policy
        self.configuration = configuration

        if configuration.compile:
            self.policy.compile()

    def update(self, phase: Phase):
        super().update(phase)

        self.policy.update(phase)

