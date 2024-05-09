from typing import Dict

import torch

from agents.utils.policy import from_cli as policy_from_cli
from .model import *


class DeepMR(DeepPolicyMachineModel):

    def __call__(self, state: State, parameters: Input) -> DeepPolicyMachineModel.Record:
        # No gradient descent based on decision on the moment
        with torch.no_grad():
            record = super().__call__(state, parameters)
            job_id = state.info['job_idx'][record.action].int()

            result = [job for job in parameters.machine.queue if job_id == job.id]

            if len(result) == 0:
                result = parameters.machine.queue[0]
            else:
                result = result[0]

            return DeepPolicyMachineModel.Record(result=result, record=record, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: Dict):
        policy_parameters = parameters['policy']
        policy_parameters['parameters']['n_actions'] = 4

        policy = policy_from_cli(policy_parameters)

        return cls(policy)
