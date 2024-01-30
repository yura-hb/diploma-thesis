import torch

import environment
import argparse

from .job_sampler import JobSampler
from dataclasses import dataclass


class UniformJobSampler(JobSampler):

    @dataclass
    class Configuration:
        # Minimum number of operations per job
        min_operations_per_job: int = None
        # Parameters of uniform distribution for sampling processing times
        processing_times: torch.Tensor = torch.tensor([1, 10])
        # Tightness factor
        tightness_factor: float = 0
        # Expected utilization rate of machines
        expected_utilization: float = 0.75
        # Seed for all random operations
        seed: int = 0

    def __init__(self, problem: environment.Problem, configuration: Configuration):
        super().__init__(problem)

        self.configuration = configuration
        self.beta = self.__make_beta__()

        torch.manual_seed(self.configuration.seed)

        self.generator = torch.Generator()

        self.arrival_time_distribution = torch.distributions.Exponential(1 / self.beta)
        self.processing_time_distribution = torch.distributions.Uniform(low=self.configuration.processing_times[0],
                                                                        high=self.configuration.processing_times[1])
        self.due_time_distribution = torch.distributions.Uniform(low=0, high=self.configuration.tightness_factor)

    def number_of_jobs(self):
        return torch.round(self.problem.timespan / self.beta)

    def sample(self, job_id: int, moment: torch.FloatType) -> environment.Job:
        work_center_idx = torch.randperm(self.problem.workcenter_count, generator=self.generator)

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
        mean, _ = job.processing_time_moments()
        tightness = 1 + self.due_time_distribution.rsample((1,))
        num_machines = self.problem.machines_per_workcenter * self.problem.workcenter_count

        return torch.round(mean * num_machines * tightness + moment)

    @staticmethod
    def add_cli_arguments(sub_parser: argparse.ArgumentParser):
        sub_parser.add_argument(
            "--min-operations-per-job",
            help="The minimum number of operations per job",
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
            default=1.0
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
            min_operations_per_job=namespace.min_operations_per_job,
            processing_times=torch.tensor(namespace.processing_times),
            tightness_factor=namespace.tightness_factor,
            expected_utilization=namespace.expected_utilization,
            seed=namespace.seed
        )

        return UniformJobSampler(problem=problem, configuration=configuration)

