
from typing import List, Dict

import torch

from agents.base.state import TensorState
from agents.utils import NNCLI, Phase, PhaseUpdatable
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .model import *
from .rule import ALL_SCHEDULING_RULES, SchedulingRule


class MultiRuleLinear(NNMachineModel, PhaseUpdatable):

    def __init__(self, rules: List[SchedulingRule], model: NNCLI, action_selector: ActionSelector):
        super().__init__()

        self.rules = rules
        self.model = model
        self.action_selector = action_selector

    def update(self, phase: Phase):
        self.phase = phase

        for module in [self.model, self.action_selector]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def __call__(self, state: State, parameters: MachineModel.Input) -> MachineModel.Record:
        values = self.values(state)
        action, _ = self.action_selector(values)
        action = torch.tensor(action, dtype=torch.long)
        rule = self.rules[action]

        return MachineModel.Record(
            result=rule(parameters.machine, parameters.now),
            state=state,
            action=action
        )

    def values(self, state: State) -> torch.FloatTensor:
        assert isinstance(state, TensorState), f"State must conform to TensorState"

        if not self.model.is_connected:
            self.__connect__(len(self.rules), self.model, state.state.shape[-1])

        return self.model(state.state)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def copy_parameters(self, other: 'MultiRuleLinear', decay: float = 1.0):
        self.model.copy_parameters(other.model, decay)

    def clone(self):
        new_model = self.model.clone()

        return MultiRuleLinear(rules=self.rules, model=new_model, action_selector=self.action_selector)

    # Utilities

    @staticmethod
    def __connect__(n_rules: int, model: NNCLI, input_shape: torch.Size):
        output_layer = NNCLI.Configuration.Linear(
            dim=n_rules,
            activation='none',
            dropout=0
        )

        model.connect(input_shape, output_layer)

    @staticmethod
    def from_cli(parameters: Dict):
        rules = parameters['rules']

        if rules == "all":
            rules = [rule() for rule in ALL_SCHEDULING_RULES.values()]
        else:
            rules = [ALL_SCHEDULING_RULES[rule]() for rule in rules]

        nn_cli = NNCLI.from_cli(parameters['nn'])

        action_selector = action_selector_from_cli(parameters['action_selector'])

        return MultiRuleLinear(rules, nn_cli, action_selector)

