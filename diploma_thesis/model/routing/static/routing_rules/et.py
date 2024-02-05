from typing import List

from environment import Machine
from environment.job import Job, ReductionStrategy
from model.routing.static.routing_rules import RoutingRule


class ETRoutingRule(RoutingRule):

    def select_machine(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        operation_times = job.operation_processing_time_in_work_center(work_center_idx, ReductionStrategy.none)

        return machines[operation_times.argmin()]
