import logging
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic

from tensordict.prototype import tensorclass

from agents.utils import Phase, PhaseUpdatable
from agents.utils.policy import Policy, PolicyRecord
from agents.utils.run_configuration import RunConfiguration
from utils import Loggable
from .state import State


Input = TypeVar('Input')
Action = TypeVar('Action')
Result = TypeVar('Result')


class Model(Generic[Input, Action, Result], Loggable, metaclass=ABCMeta):

    @tensorclass
    class Record:
        result: Result
        record: PolicyRecord | None

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, state: State, parameters: Input) -> Record:
        pass

    def compile(self):
        pass

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict: dict):
        pass


class DeepPolicyModel(Model[Input, Action, Result], PhaseUpdatable, metaclass=ABCMeta):

    def __init__(self, policy: Policy[Input]):
        super().__init__()

        self.policy = policy

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        if isinstance(self.policy, Loggable):
            self.policy.with_logger(logger)

        return self

    def __call__(self, state: State, parameters: Input) -> PolicyRecord:
        state.memory = parameters.memory

        return self.policy.select(state)

    def update(self, phase: Phase):
        super().update(phase)

        self.policy.update(phase)

    def configure(self, configuration: RunConfiguration):
        self.policy.configure(configuration)

    def state_dict(self):
        return self.policy.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.policy.load_state_dict(state_dict, strict=False)
