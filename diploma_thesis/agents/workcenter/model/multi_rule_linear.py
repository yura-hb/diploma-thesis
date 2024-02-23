from agents.utils import DeepRule
from .model import *
from .rule import ALL_ROUTING_RULES, RoutingRule


class MultiRuleLinear(NNWorkCenterModel, DeepRule):

    def __call__(self, state: State, parameters: WorkCenterModel.Input) -> WorkCenterModel.Record:
        return DeepRule.__call__(self, state, parameters)

    @classmethod
    def all_rules(cls):
        return ALL_ROUTING_RULES

    def make_result(
         self, rule: RoutingRule, parameters: WorkCenterModel.Input, state: State, action: Action
    ) -> WorkCenterModel.Record:
        return WorkCenterModel.Record(result=rule(state, parameters), state=state, action=action)
