from .scheduling_rule import *


class LPTSchedulingRule(SchedulingRule):
    """
    Longest Processing Time rule, i.e. selects jobs, in which current operation has the largest operation time
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        values = torch.FloatTensor([job.current_operation_processing_time_on_machine for job in machine.queue])
        idx = torch.argmax(values)

        return machine.queue[idx]
