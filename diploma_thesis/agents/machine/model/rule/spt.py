from .scheduling_rule import *


class SPTSchedulingRule(SchedulingRule):
    """
    Shortest Processing Time rule, i.e. selects jobs, in which current operation has the smallest operation time
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        values = [job.current_operation_processing_time_on_machine for job in machine.queue]
        idx = torch.argmin(values)

        return machine.queue[idx]
