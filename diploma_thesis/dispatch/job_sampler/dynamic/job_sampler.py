
from typing import Callable, Tuple

import simpy
import torch

from environment import Configuration, Job
from dispatch.sampler import Sampler
from dispatch.job_sampler import JobSampler as JSampler


class JobSampler(JSampler):

    def __init__(self, problem: Configuration):
        super().__init__(problem)

        self._number_of_jobs = None
        self._processing_time_sampler: Callable[[Tuple[int]], torch.FloatTensor] = None
        self._step_sampler: Callable[[int], torch.LongTensor] = None
        self._due_time_sampler: Callable[['Job', int], torch.FloatTensor] = None
        self.samplers: [Sampler] = []
        self.arrival_time_sampler = None

        self._jobs = []

    def connect(self, generator: torch.Generator):
        for sampler in self.samplers:
            sampler.connect(generator)

    def number_of_jobs(self):
        return self._number_of_jobs

    def sample(self, job_id: int, initial_work_center_idx: int, moment: torch.FloatType) -> Job:
        step_idx = self._step_sampler(initial_work_center_idx)

        shape = (len(step_idx), self.problem.machines_per_work_center)

        processing_times = self._processing_time_sampler(shape)

        job = Job(id=job_id, step_idx=step_idx, processing_times=processing_times)

        event = Job.Event(kind=Job.Event.Kind.creation, moment=moment)
        job = job.with_event(event)

        due_at = self._due_time_sampler(job, moment)

        return job.with_due_at(due_at)

    def sample_next_arrival_time(self) -> torch.FloatTensor:
        return self.arrival_time_sampler()

    def store(self, sampler: Sampler):
        self.samplers += [sampler]
