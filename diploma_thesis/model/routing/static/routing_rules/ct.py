
from typing import List

import torch

from environment import Machine
from environment.job import Job, ReductionStrategy
from model.routing.static.routing_rules import RoutingRule


class CTRoutingRule(RoutingRule):
    """
    Earliest Completion Time rule
    """

    def select_machine(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        cumulative_processing_times = torch.FloatTensor(
            [machine.cumulative_processing_time for machine in machines]
        ) + job.operation_processing_time_in_work_center(work_center_idx, ReductionStrategy.none)

        idx = torch.argmin(cumulative_processing_times)

        return machines[idx]
