from environment.job import Job, ReductionStrategy
from routing_rules import WorkCenterState, RoutingRule


class ETRoutingRule(RoutingRule):

    def select_machine(self, job: Job, state: WorkCenterState) -> 'Machine':
        operation_times = job.operation_processing_time_in_work_center(state.work_center_idx, ReductionStrategy.none)

        return state.machines[operation_times.argmin()]
