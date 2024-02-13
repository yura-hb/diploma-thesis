
from typing import List, Dict

import torch

from agents.utils import NNCLI, Phase, PhaseUpdatable
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .model import *
from .rule import ALL_SCHEDULING_RULES
from .rule import SchedulingRule


class MultiRuleLinear(Model, PhaseUpdatable):

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
        if not self.model.is_connected:
            self.__connect__(state.shape)

        distribution = self.model(state, parameters)

        action, _ = self.action_selector(distribution)

        rule = self.rules[action]

        return MachineModel.Record(
            result=rule(parameters.machine, parameters.now),
            state=state,
            action=action
        )

    def __connect__(self, input_shape: torch.Size):
        output_layer = NNCLI.Configuration.Linear(
            dim=len(self.rules),
            activation='none',
            dropout=0
        )

        self.model.connect(input_shape, output_layer)

    @staticmethod
    def from_cli(parameters: Dict):
        rules = parameters['rules']

        if rules == "all":
            rules = [rule() for rule in ALL_SCHEDULING_RULES.values()]
        else:
            rules = [ALL_SCHEDULING_RULES[rule]() for rule in rules]

        nn_cli = NNCLI.Configuration.from_cli(parameters['nn'])

        action_selector = action_selector_from_cli(parameters['action_selector'])

        return MultiRuleLinear(rules, nn_cli, action_selector)


