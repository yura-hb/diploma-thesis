
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


class Model(Loggable, Generic[Input, Action, Result], metaclass=ABCMeta):

    @tensorclass
    class Record:
        result: Result
        record: PolicyRecord | None

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
        self.memory = dict()

    def __call__(self, state: State, parameters: Input) -> PolicyRecord:
        if self.policy.is_recurrent:
            key = self.memory_key(parameters)

            assert key is not None, 'Expect that key definition for recurrent policy'

            state.memory = self.memory.get(key)
            record, memory = self.policy.select(state)

            self.memory[key] = memory

            return record

        return self.policy.select(state)

    @classmethod
    def memory_key(cls, parameters: Input) -> None | str:
        return None

    def update(self, phase: Phase):
        super().update(phase)

        self.policy.update(phase)

    def configure(self, configuration: RunConfiguration):
        self.policy.configure(configuration)

    def state_dict(self):
        return self.policy.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.policy.load_state_dict(state_dict)