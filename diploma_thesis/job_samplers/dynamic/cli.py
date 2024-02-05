
from dataclasses import dataclass
from typing import Tuple
from environment import Configuration, JobSampler
import simpy
from .builder import Builder


class CLI:

    @dataclass
    class Configuration:
        processing_times: Tuple[int, int]
        tightness: float
        uneveness: float
        realistic_variance: int
        expected_utilization: float
        even_arrival_time: float
        seed: int

        @staticmethod
        def from_cli_arguments(args: dict):
            return CLI.Configuration(
                processing_times=args['processing_times'],
                tightness=args['tightness'],
                uneveness=args.get('uneveness'),
                realistic_variance=args.get('realistic_variance'),
                expected_utilization=args.get('expected_utilization'),
                even_arrival_time=args.get('even_arrival_time'),
                seed=args['seed']
            )

    @staticmethod
    def from_cli_arguments(problem: Configuration,
                           environment: simpy.Environment,
                           parameters: dict) -> JobSampler:
        configuration = CLI.Configuration.from_cli_arguments(parameters)

        builder = Builder(problem, environment, configuration.seed)

        if realistic_variance := configuration.realistic_variance:
            builder.with_uniform_processing_times_and_realistic_variance(
                configuration.processing_times, realistic_variance
            )
        else:
            builder.with_uniform_processing_times(configuration.processing_times)

        builder.with_uniform_step_generation(configuration.uneveness)

        builder.with_uniform_due_time(configuration.tightness)

        if expected_utilization := configuration.expected_utilization:
            builder.with_exponential_arrival_time_from_processing_time(
                configuration.processing_times, expected_utilization
            )
        elif even_arrival_time := configuration.even_arrival_time:
            builder.with_even_arrival_time(even_arrival_time)
        else:
            raise ValueError("Either expected_utilization or even_arrival_time must be provided")

        return builder.sampler, configuration
