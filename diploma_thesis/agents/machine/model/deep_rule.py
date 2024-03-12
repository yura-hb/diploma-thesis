from typing import Dict

import torch

from agents.utils.policy import from_cli as policy_from_cli
from .model import *


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

        return cls(policy)
