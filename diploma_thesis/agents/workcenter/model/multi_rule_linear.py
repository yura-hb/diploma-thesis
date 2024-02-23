from typing import List

from agents.utils import DeepRule, NNCLI, ActionSelector
from .model import *
from .rule import ALL_ROUTING_RULES, RoutingRule


class MultiRuleLinear(NNWorkCenterModel, DeepRule):

    def __init__(self, rules: List[RoutingRule], model: NNCLI, action_selector: ActionSelector):
        NNWorkCenterModel.__init__(self)
        DeepRule.__init__(self, rules, model, action_selector)

    def __call__(self, state: State, parameters: WorkCenterModel.Input) -> WorkCenterModel.Record:
        return DeepRule.__call__(self, state, parameters)

    @classmethod
    def all_rules(cls):
        return ALL_ROUTING_RULES

    def make_result(
         self, rule: RoutingRule, parameters: WorkCenterModel.Input, state: State, action: Action
    ) -> WorkCenterModel.Record:
        return WorkCenterModel.Record(
            result=rule(job=parameters.job, work_center=parameters.work_center),
            state=state,
            action=action
        )

    def clone(self):
        new_model = self.model.clone()

        return MultiRuleLinear(rules=self.rules, model=new_model, action_selector=self.action_selector)
