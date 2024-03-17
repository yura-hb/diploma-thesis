from typing import Dict

import torch

from agents.base import Graph
from agents.utils.policy import from_cli as policy_from_cli
from .model import *


class DeepRule(DeepPolicyMachineModel):

    def __call__(self, state: State, parameters: Input) -> DeepPolicyMachineModel.Record:
        # No gradient descent based on decision on the moment
        with torch.no_grad():
            record = super().__call__(state, parameters)

            job_index_map = state.graph.data[Graph.JOB_INDEX_MAP]
            store = state.graph.data[Graph.OPERATION_KEY] \
                if Graph.OPERATION_KEY in state.graph.data.keys() else state.graph.data
            target = store[Graph.TARGET_KEY][:job_index_map.shape[0]]

            job_id = job_index_map[target, 0][record.action]

            result = [job for job in parameters.machine.queue if job_id == job.id][0]

            return DeepPolicyMachineModel.Record(result=result, record=record, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: Dict):
        policy_parameters = parameters['policy']
        policy = policy_from_cli(policy_parameters)

        return cls(policy)
