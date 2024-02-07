from .scheduling_rule import *


class NPTSchedulingRule(SchedulingRule):
    """
    Next Processing Time rule, i.e. selects jobs, in which the processing time of the next operation in the job is
    the smallest
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        values = torch.FloatTensor([job.next_operation_processing_time() for job in machine.queue])
        idx = torch.argmin(values)

        return machine.queue[idx]
