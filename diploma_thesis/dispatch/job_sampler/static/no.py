import torch

from dispatch.job_sampler import JobSampler
from environment import Job, Configuration


class No(JobSampler):

    def __init__(self, problem: Configuration):
        super().__init__(problem)

        self._last_created_job_id = 0

    def number_of_jobs(self):
        return 0

    def sample(self, job_id: int, initial_work_center_idx: int, moment: float) -> Job:
        self._last_created_job_id = job_id

        processing_times = torch.ones((self.problem.work_center_count, self.problem.machines_per_work_center))
        step_idx = torch.arange(0, self.problem.work_center_count, dtype=torch.int32)

        step_idx[initial_work_center_idx] = 0
        step_idx[0] = initial_work_center_idx

        job = Job(id=job_id, step_idx=step_idx, processing_times=processing_times, batch_size=[])

        event = Job.Event(kind=Job.Event.Kind.creation, moment=moment)
        job = job.with_event(event)

        return job.with_due_at(100000)

    def sample_next_arrival_time(self) -> float:
        return 0.0

    @staticmethod
    def from_cli(parameters: dict, problem: Configuration):
        return No(problem=problem)
