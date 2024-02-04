import torch

import environment
import argparse

from .job_sampler import JobSampler
from dataclasses import dataclass


class UniformJobSampler(JobSampler):
    @dataclass
    class Configuration:
        # A parameter for categorical distribution which determines the maximum number of steps
        # dropped from generated sequence
        job_unevenness_factor: int = 0
        # Parameters of uniform distribution for sampling processing times
        processing_times: torch.Tensor = torch.tensor([1, 10])
        # Tightness factor
        tightness_factor: float = 0
        # Expected utilization rate of machines
        expected_utilization: float = 0.75
        # Seed for all random operations
        seed: int = 0

        @property
        def is_uneven_job_creation_expected(self):
            return self.job_unevenness_factor > 0

    def __init__(self, problem: environment.Problem, configuration: Configuration):
        super().__init__(problem)

        self.configuration = configuration
        self.beta = self.__make_beta__()
        self._number_of_jobs = torch.round(self.problem.timespan / self.beta)

        torch.manual_seed(self.configuration.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.configuration.seed)

        self.arrival_time_distribution = torch.distributions.Exponential(1 / self.beta)
        self.processing_time_distribution = torch.distributions.Uniform(low=self.configuration.processing_times[0],
                                                                        high=self.configuration.processing_times[1])
        self.due_time_distribution = torch.distributions.Uniform(low=0, high=self.configuration.tightness_factor)

        if self.configuration.is_uneven_job_creation_expected:
            self.uneveness_distribution = torch.distributions.Uniform(
                low=0, high=self.configuration.job_unevenness_factor
            )

    def number_of_jobs(self):
        return self._number_of_jobs

    def sample(self, job_id: int, initial_work_center_idx: int, moment: torch.FloatType) -> environment.Job:
        work_center_idx = torch.randperm(self.problem.workcenter_count, generator=self.generator)

        if initial_work_center_idx is not None:
            initial_work_center_idx = torch.LongTensor([initial_work_center_idx])
            work_center_idx = work_center_idx[work_center_idx != initial_work_center_idx]
            work_center_idx = torch.hstack([initial_work_center_idx, work_center_idx])

        if self.configuration.is_uneven_job_creation_expected:
            cut_off = self.uneveness_distribution.sample((1,))
            cut_off = int(cut_off)

            if cut_off > 0:
                work_center_idx = work_center_idx[:-cut_off]

        shape = (len(work_center_idx), self.problem.machines_per_workcenter)

        processing_times = self.processing_time_distribution.sample(shape)
        processing_times = torch.round(processing_times)

        job = environment.Job(id=job_id, step_idx=work_center_idx, processing_times=processing_times)

        job = job.with_event(
            environment.Job.Event(
                kind=environment.Job.Event.Kind.creation,
                moment=moment,
            )
        )

        due_at = self.__sample_due_time__(job=job, moment=moment)

        return job.with_due_at(due_at)

    def sample_next_arrival_time(self) -> torch.FloatTensor:
        return self.arrival_time_distribution.sample()

    def __make_beta__(self):
        distance = self.configuration.processing_times[1] - self.configuration.processing_times[0]
        mean = torch.Tensor([self.configuration.processing_times[0] + distance / 2])

        return mean / (self.problem.machines_per_workcenter * self.configuration.expected_utilization)

    def __sample_due_time__(self, job: environment.Job, moment: int):
        # Take mean over all processing times of the job
        mean, _ = job.processing_time_moments(reduction_strategy=environment.JobReductionStrategy.none)
        tightness = 1. + self.due_time_distribution.rsample((1,))
        num_machines = job.step_idx.shape[0]

        return torch.round(mean * num_machines * tightness + moment)

    @staticmethod
    def add_cli_arguments(sub_parser: argparse.ArgumentParser):
        sub_parser.add_argument(
            "--job_unevenness_factor",
            help="A parameter for categorical distribution"
                 "which determines the maximum number of steps dropped from generated sequence",
            type=int,
            default=0
        )

        sub_parser.add_argument(
            "--processing-times",
            help="The range of processing times",
            type=float,
            nargs=2,
            default=[1, 10]
        )

        sub_parser.add_argument(
            "--tightness-factor",
            help="The tightness factor",
            type=float,
            default=0.1
        )

        sub_parser.add_argument(
            "--expected-utilization",
            help="The expected utilization rate of machines",
            type=float,
            default=0.75
        )

        sub_parser.add_argument(
            "--seed",
            help="The seed for all random operations",
            type=int,
            default=0
        )

    @staticmethod
    def from_cli_arguments(problem: environment.Problem, namespace: argparse.Namespace) -> 'UniformJobSampler':
        configuration = UniformJobSampler.Configuration(
            job_unevenness_factor=namespace.job_unevenness_factor,
            processing_times=torch.tensor(namespace.processing_times),
            tightness_factor=namespace.tightness_factor,
            expected_utilization=namespace.expected_utilization,
            seed=namespace.seed
        )

        return UniformJobSampler(problem=problem, configuration=configuration)

