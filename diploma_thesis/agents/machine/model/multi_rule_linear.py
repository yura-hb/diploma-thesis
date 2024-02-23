from agents.utils import DeepRule
from .model import *
from .rule import ALL_SCHEDULING_RULES, SchedulingRule


class MultiRuleLinear(NNMachineModel, DeepRule):

    def __call__(self, state: State, parameters: MachineModel.Input) -> MachineModel.Record:
        return DeepRule.__call__(self, state, parameters)

    @classmethod
    def all_rules(cls):
        return ALL_SCHEDULING_RULES

    def make_result(
        self, rule: SchedulingRule, parameters: MachineModel.Input, state: State, action: Action
    ) -> MachineModel.Record:
        return MachineModel.Record(result=rule(state, parameters), state=state, action=action)
