import torch

from .routing_rule import *


class CTRoutingRule(RoutingRule):
    """
    Earliest Completion Time rule
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        return torch.FloatTensor(
            [machine.cumulative_processing_time for machine in work_center.machines]
        ) + job.operation_processing_time_in_work_center(work_center.work_center_idx, JobReductionStrategy.none)
