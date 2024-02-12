from .scheduling_rule import *


class LWRKSPTSchedulingRule(SchedulingRule):
    """
    Least Work Remaining + Shortest Processing Time rule,
        i.e. selects jobs, in which satisfy both criteria (for reference check lwrk.py and spt.py)
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        next_remaining_processing_time = torch.FloatTensor([
            job.next_remaining_processing_time(self.reduction_strategy) for job in machine.queue
        ])
        idx = torch.argmin(2 * processing_times + next_remaining_processing_time)

        return machine.queue[idx]
