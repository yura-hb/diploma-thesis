from typing import Tuple

import simpy
import torch

from environment import Configuration, Job, JobReductionStrategy
from dispatch.sampler import NumericSampler, Permutation
from .job_sampler import JobSampler


class Builder:

    def __init__(self, problem: Configuration):
        self.problem = problem
        self.job_sampler = JobSampler(problem)

    def with_step_generation(self, sampler: Permutation):
        def sample(initial_work_center_idx: int):
            sampler.update(initial_work_center_idx)

            return sampler.sample([self.problem.work_center_count])

        self.job_sampler._step_sampler = sample
        self.job_sampler.store(sampler)

        return self

    def with_processing_time_distribution(self, sampler: NumericSampler):
        def sample(shape: Tuple[int]) -> torch.FloatTensor:
            times = sampler.sample(shape)
            times = torch.round(times)

            return times

        self.job_sampler._processing_time_sampler = sample
        self.job_sampler.store(sampler)

        return self

    def with_uniform_due_time(self, sampler: NumericSampler):
        def sample(job: Job, moment: torch.FloatTensor) -> torch.FloatTensor:
            # Take mean over all processing times of the job
            weight = job.step_idx.shape[0]
            weight *= job.processing_time_moments(reduction_strategy=JobReductionStrategy.none)[0]

            tightness = 1. + sampler.sample((1,))

            return torch.round(weight * tightness + moment)

        self.job_sampler._due_time_sampler = sample
        self.job_sampler.store(sampler)

        return self

    def with_job_arrival_time(self, sampler: NumericSampler, n_jobs: int = None):
        def sample() -> torch.FloatTensor:
            return sampler.sample((1,))

        self.job_sampler.arrival_time_sampler = sample
        self.job_sampler._number_of_jobs = n_jobs
        self.job_sampler.store(sampler)

        return self

    @property
    def sampler(self) -> JobSampler:
        return self.job_sampler
