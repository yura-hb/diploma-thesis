from typing import Tuple

import simpy
import torch

from environment import Problem, Job, JobReductionStrategy
from .job_sampler import Sampler


class Builder:

    def __init__(self, problem: Problem, environment: simpy.Environment, seed: int = 0):
        self.problem = problem
        self.environment = environment
        self.job_sampler = Sampler(problem, environment)

        torch.manual_seed(seed)

    def with_uniform_step_generation(self, uneveness: float = 0):
        """
        Args:
            uneveness: Maximum number of jobs dropped from generated sequence
        """
        def sample(initial_work_center_idx: int):
            step_idx = torch.randperm(self.job_sampler.problem.workcenter_count)

            if initial_work_center_idx is not None:
                initial_work_center_idx = torch.LongTensor([initial_work_center_idx])
                step_idx = step_idx[step_idx != initial_work_center_idx]
                step_idx = torch.hstack([initial_work_center_idx, step_idx])

            if uneveness > 0:
                cut_off = torch.randint(0, uneveness, (1,))
                cut_off = int(cut_off)

                if cut_off > 0:
                    step_idx = step_idx[:-cut_off]

            return step_idx

        self.job_sampler._step_sampler = sample

        return self

    def with_uniform_processing_times(self, processing_times: Tuple[int, int]):
        """
        Args:
            processing_times: Minimum and maximum processing time for job processing time
        """
        distribution = torch.distributions.Uniform(low=processing_times[0], high=processing_times[1])

        def sample(shape: Tuple[int]) -> torch.FloatTensor:
            times = distribution.sample(shape)
            times = torch.round(times)

            return times

        self.job_sampler._processing_time_sampler = sample

        return self

    def with_uniform_processing_times_and_realistic_variance(self, processing_times: Tuple[int, int], variance: int):
        """
        Args:
            processing_times: Minimum and maximum processing time for job processing time
            variance: The variance of noise added to the processing times
        """
        assert variance > 0, "Variance must be positive"

        distribution = torch.distributions.Uniform(low=processing_times[0], high=processing_times[1])
        noise = torch.distributions.Normal(loc=0, scale=torch.sqrt(variance))

        def sample(shape: Tuple[int]) -> torch.FloatTensor:
            times = distribution.sample(shape) + noise.sample(shape)
            times = times.clip(processing_times[0], processing_times[1])
            times = torch.round(times)

            return times

        self.job_sampler._processing_time_sampler = sample

        return self

    def with_uniform_due_time(self, tightness: float):
        """
        Args:
            tightness: Value determining how tight the due time is to the expected processing time
        """
        distribution = torch.distributions.Uniform(low=0, high=tightness)

        def sample(job: Job, moment: torch.FloatTensor) -> torch.FloatTensor:
            # Take mean over all processing times of the job
            mean, _ = job.processing_time_moments(reduction_strategy=JobReductionStrategy.none)
            tightness = 1. + distribution.rsample((1,))
            num_machines = job.step_idx.shape[0]

            return torch.round(mean * num_machines * tightness + moment)

        self.job_sampler._due_time_sampler = sample

        return self

    def with_exponential_arrival_time_from_processing_time(
        self, processing_times: Tuple[int, int], expected_utilization: float
    ):
        """

        Args:
            processing_times: Minimum and maximum processing time for job processing time
            expected_utilization: Expected utilization of shopfloor in range [0, 1]
        """
        expected_utilization = torch.FloatTensor([expected_utilization])
        expected_utilization = expected_utilization.clip(0, 1)

        distance = processing_times[1] - processing_times[0]
        mean = torch.Tensor([processing_times[0] + distance / 2])
        beta = mean / (self.problem.machines_per_workcenter * expected_utilization)

        return self.with_exponential_arrival_time(rate=1 / beta)

    def with_exponential_arrival_time(self, rate: torch.FloatTensor):
        """
        Args:
            rate: Rate of exponential distribution
        """
        distribution = torch.distributions.Exponential(rate=rate)

        def sample() -> torch.FloatTensor:
            return distribution.sample()

        self.job_sampler.arrival_time_sampler = sample
        self.job_sampler._number_of_jobs = torch.round(self.problem.timespan * rate)

        return self

    def with_even_arrival_time(self, interval: float):
        """
        Args:
            interval: Time between each job arrival
        """
        def sample() -> torch.FloatTensor:
            return torch.FloatTensor([interval])

        self.job_sampler.arrival_time_sampler = sample
        self.job_sampler._number_of_jobs = torch.round(self.problem.timespan / interval)

        return self

    @property
    def sampler(self) -> Sampler:
        return self.job_sampler
