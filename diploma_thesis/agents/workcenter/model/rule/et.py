from .routing_rule import *


class ETRoutingRule(RoutingRule):

    def __call__(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        operation_times = job.operation_processing_time_in_work_center(work_center_idx, JobReductionStrategy.none)

        return machines[operation_times.argmin()]
