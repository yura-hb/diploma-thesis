from .scheduling_rule import *


class GP2SchedulingRule(SchedulingRule):
    """
    Genetic Programming 2 scheduling rule. Taken from external/PhD-Thesis-Projects/FJSP/sequencing.py
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        operation_processing_time = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine.queue]
        )
        next_remaining_processing_time = torch.FloatTensor(
            [job.next_remaining_processing_time(self.reduction_strategy) for job in machine.queue]
        )
        winq = torch.FloatTensor([
            machine.shop_floor.work_in_next_queue(job) for job in machine.queue
        ])
        avlm = torch.FloatTensor([
            machine.shop_floor.average_waiting_in_next_queue(job) for job in machine.queue
        ])
        number_of_jobs_in_queue = len(machine.queue)

        s1 = number_of_jobs_in_queue * (operation_processing_time - 1)

        s2 = operation_processing_time
        s2 = s2 + next_remaining_processing_time * torch.max(
            torch.vstack([operation_processing_time, winq]), dim=0
        )[0]

        s3 = torch.max(torch.vstack([winq, winq + number_of_jobs_in_queue]), dim=0)[0]

        s4 = avlm + 1 + torch.max(
            torch.vstack(
                [next_remaining_processing_time,
                 torch.ones_like(next_remaining_processing_time) * (number_of_jobs_in_queue - 1)]
            ), dim=0
        )[0]
        s4 *= torch.max(torch.vstack([winq, next_remaining_processing_time]), dim=0)[0]

        values = s1 * s2 + s3 * s4
        idx = torch.argmin(values)

        return machine.queue[idx]
