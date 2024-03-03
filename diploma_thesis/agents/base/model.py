
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic

from tensordict.prototype import tensorclass

from agents.utils.policy import Policy, PolicyRecord
from utils import Loggable

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


class DeepPolicyModel(Model[Input, State, Action, Result], metaclass=ABCMeta):

    def __init__(self, policy: Policy[Input]):
        super().__init__()

        self.policy = policy

    def __call__(self, state: State, parameters: Input) -> Model.Record:
        return self.policy(state, parameters)

