from environment.job import Job
from routing_rules import WorkCenterState, RoutingRule


class ETRoutingRule(RoutingRule):

    def select_machine(self, job: Job, state: WorkCenterState) -> int:
        operation_times = job.operation_processing_time_in_work_center(state.work_center_idx, Job.ReductionStrategy.none)

        return operation_times.index(min(operation_times))
