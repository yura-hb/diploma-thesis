
import torch

from environment.job import Job, ReductionStrategy
from routing_rules import WorkCenterState, RoutingRule


class CTRoutingRule(RoutingRule):
    """
    Earliest Completion time rule
    """

    def select_machine(self, job: Job, state: WorkCenterState) -> 'Machine':
        cumulative_processing_times = torch.FloatTensor(
            [machine.cumulative_processing_time for machine in state.machines]
        ) + job.operation_processing_time_in_work_center(state.work_center_idx, ReductionStrategy.none)

        idx = torch.argmin(cumulative_processing_times)

        return state.machines[idx]
