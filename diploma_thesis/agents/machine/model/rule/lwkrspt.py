from .scheduling_rule import *


class LWRKSPTSchedulingRule(SchedulingRule):
    """
    Least Work Remaining + Shortest Processing Time rule,
        i.e. selects jobs, in which satisfy both criteria (for reference check lwrk.py and spt.py)
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        values = torch.FloatTensor([
            job.remaining_processing_time() for job in machine.queue
        ])
        idx = torch.argmin(values)

        return machine.queue[idx]
