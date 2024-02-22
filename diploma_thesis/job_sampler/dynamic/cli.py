
from dataclasses import dataclass
from typing import Tuple

import simpy

from environment import JobSampler, Configuration as Problem
from sampler import (NumericSampler, Permutation, Exponential, Constant,
                     numeric_sampler_from_cli, permutation_sampler_from_cli)
from .builder import Builder


@dataclass
class Template:
    pass


@dataclass
class Custom(Template):
    sampler: NumericSampler


@dataclass
class ExpectedUtilization:
    expected_utilization: float


@dataclass
class EvenArrivalTime:
    time: float


class CLI:

    @dataclass
    class Configuration:
        processing_times: NumericSampler
        permutation: Permutation
        due_time: NumericSampler
        job_arrival_time_on_machine: NumericSampler
        n_jobs: int

    @staticmethod
    def from_cli(parameters: dict, problem: Problem, environment: simpy.Environment) -> JobSampler:
        configuration = CLI.configuration_from_cli_arguments(problem, parameters)

        builder = Builder(problem, environment)

        builder.with_processing_time_distribution(configuration.processing_times)
        builder.with_step_generation(configuration.permutation)
        builder.with_uniform_due_time(configuration.due_time)
        builder.with_job_arrival_time(configuration.job_arrival_time_on_machine, configuration.n_jobs)

        return builder.sampler

    @staticmethod
    def configuration_from_cli_arguments(problem: Problem, parameters: dict):
        processing_times = numeric_sampler_from_cli(parameters['processing_times'])
        job_arrival_time_on_machine, n_jobs = CLI.arrival_time_sampler_from_cli(
            problem, processing_times.mean, parameters['job_arrival_time_on_machine']
        )

        return CLI.Configuration(
            processing_times=processing_times,
            permutation=permutation_sampler_from_cli(parameters['permutation']),
            due_time=numeric_sampler_from_cli(parameters['due_time']),
            job_arrival_time_on_machine=job_arrival_time_on_machine,
            n_jobs=parameters.get('n_jobs'),
        )

    @staticmethod
    def arrival_time_sampler_from_cli(
         problem: Problem, mean_processing_time: float, parameters: dict
    ) -> Tuple[NumericSampler, int]:
        key = parameters['kind']
        parameters = parameters['parameters']
        number_of_machines = problem.machines_per_work_center
        sampler = None

        match key:
            case 'sampler':
                sampler = numeric_sampler_from_cli(parameters['sampler'])
            case 'expected_utilization':
                arrival_time = mean_processing_time / (number_of_machines * parameters['value'])

                sampler = Exponential(arrival_time)
            case 'even_arrival_time':
                sampler = Constant(parameters['value'] / number_of_machines)
            case _:
                raise ValueError(f"Unknown kind {parameters['kind']}")

        return sampler, parameters.get('n_jobs')
