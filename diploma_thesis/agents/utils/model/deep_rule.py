
import torch

from abc import abstractmethod, ABCMeta

from typing import List, Dict, TypeVar, Generic

from agents.base.state import TensorState
from agents.utils import NNCLI, Phase, PhaseUpdatable
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli

State = TypeVar('State')
Rule = TypeVar('Rule')
Input = TypeVar('Input')
Record = TypeVar('Record')


class DeepRule(Generic[Rule, Input, Record], PhaseUpdatable, metaclass=ABCMeta):

    def __init__(self, rules: List[Rule], model: NNCLI, action_selector: ActionSelector):
        super().__init__()

        self.rules = rules
        self.model = model
        self.action_selector = action_selector

    # Overridable

    @classmethod
    @abstractmethod
    def all_rules(cls):
        pass

    @classmethod
    @abstractmethod
    def idle_rule(cls):
        pass

    @abstractmethod
    def make_result(self, rule: Rule, parameters: Input, state: State, action) -> Record:
        pass

    @abstractmethod
    def clone(self):
        pass

    # Main

    def update(self, phase: Phase):
        self.phase = phase

        for module in [self.model, self.action_selector]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def __call__(self, state: State, parameters: Input) -> Record:
        values = self.values(state).view(-1)
        action, _ = self.action_selector(values)
        action = torch.tensor(action, dtype=torch.long)
        rule = self.rules[action]

        return self.make_result(rule, parameters, state, action)

    def values(self, state: State) -> torch.FloatTensor:
        assert isinstance(state, TensorState), f"State must conform to TensorState"

        if not self.model.is_connected:
            self.__connect__(len(self.rules), self.model, state.state.shape)

        tensor = torch.atleast_2d(state.state)

        return self.model(tensor)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def copy_parameters(self, other: 'DeepRule', decay: float = 1.0):
        self.model.copy_parameters(other.model, decay)

    # Utilities

    @staticmethod
    def __connect__(n_rules: int, model: NNCLI, input_shape: torch.Size):
        output_layer = NNCLI.Configuration.Linear(dim=n_rules, activation='none', dropout=0)

        model.connect(input_shape, output_layer)

    @classmethod
    def from_cli(cls, parameters: Dict):
        rules = parameters['rules']

        all_rules = cls.all_rules()

        if rules == "all":
            rules = [rule() for rule in all_rules.values()]
        else:
            rules = [all_rules[rule]() for rule in rules]

        if parameters.get('idle', False):
            rules = [cls.idle_rule()] + rules

        nn_cli = NNCLI.from_cli(parameters['model'])

        action_selector = action_selector_from_cli(parameters['action_selector'])

        return cls(rules, nn_cli, action_selector)
