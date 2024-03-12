
import torch

from typing import Dict, List

from agents.utils.policy import from_cli as policy_from_cli
from .model import *
from .rule import SchedulingRule, ALL_SCHEDULING_RULES, IdleSchedulingRule


class DeepRule(DeepPolicyMachineModel):

    def __call__(self, state: State, parameters: Input) -> DeepPolicyMachineModel.Record:
        # No gradient descent based on decision on the moment
        with torch.no_grad():
            record = self.policy(state, parameters)
            result = parameters.machine.queue[record.action.item()]

            return DeepPolicyMachineModel.Record(result=result, record=record, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: Dict):
        policy_parameters = parameters['policy']
        policy = policy_from_cli(policy_parameters)

        configuration = DeepPolicyModel.Configuration.from_cli(parameters)

        return cls(policy, configuration)
