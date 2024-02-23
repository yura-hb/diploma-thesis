from typing import List

from agents.utils import DeepRule, NNCLI, ActionSelector
from .model import *
from .rule import ALL_SCHEDULING_RULES, SchedulingRule


class MultiRuleLinear(NNMachineModel, DeepRule):

    def __init__(self, rules: List[SchedulingRule], model: NNCLI, action_selector: ActionSelector):
        NNMachineModel.__init__(self)
        DeepRule.__init__(self, rules, model, action_selector)

    def __call__(self, state: State, parameters: MachineModel.Input) -> MachineModel.Record:
        return DeepRule.__call__(self, state, parameters)

    @classmethod
    def all_rules(cls):
        return ALL_SCHEDULING_RULES

    def make_result(
        self, rule: SchedulingRule, parameters: MachineModel.Input, state: State, action: Action
    ) -> MachineModel.Record:
        return MachineModel.Record(
            result=rule(machine=parameters.machine, now=parameters.now),
            state=state,
            action=action
        )

    def clone(self):
        new_model = self.model.clone()

        return MultiRuleLinear(rules=self.rules, model=new_model, action_selector=self.action_selector)

