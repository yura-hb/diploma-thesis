import torch

from typing import Dict, List

from agents.utils import DiscreteAction
from agents.utils.policy import from_cli as policy_from_cli
from .model import *
from .rule import ALL_ROUTING_RULES, RoutingRule, IdleRoutingRule


class DeepMultiRule(DeepPolicyWorkCenterModel):

    def __init__(self, rules: List[RoutingRule], policy: Policy[WorkCenterInput]):
        super().__init__(policy)

        self.rules = rules

    def __call__(self, state: State, parameters: WorkCenterModel.Input) -> WorkCenterModel.Record:
        # No gradient descent based on decision on the moment
        with torch.no_grad():
            record = self.policy(state, parameters)
            result = self.rules[record.action.item()](parameters.machine, parameters.now)

            return DeepPolicyWorkCenterModel.Record(result=result, record=record, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: Dict):
        rules = parameters['rules']

        all_rules = ALL_ROUTING_RULES

        if rules == "all":
            rules = [rule() for rule in all_rules.values()]
        else:
            rules = [all_rules[rule]() for rule in rules]

        if parameters.get('idle', False):
            rules = [IdleRoutingRule()] + rules

        policy_parameters = parameters['policy']
        policy_parameters['parameters']['n_actions'] = len(rules)

        policy = policy_from_cli(policy_parameters)

        return cls(rules, policy)
