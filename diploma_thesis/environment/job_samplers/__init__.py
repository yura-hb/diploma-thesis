
import environment

from .uniform_job_sampler import UniformJobSampler
from .job_sampler import JobSampler

import argparse

destination_key = "sampler"

key_to_sampler = {
    "uniform_sampler": UniformJobSampler
}


def add_cli_arguments(parser: argparse.ArgumentParser):
    sub_parsers = parser.add_subparsers(dest=destination_key)

    for key, sampler in key_to_sampler.items():
        sub_parser = sub_parsers.add_parser(name="uniform_sampler")

        sampler.add_cli_arguments(sub_parser)


def from_cli_arguments(problem: environment.Problem, namespace: argparse.Namespace) -> 'JobSampler':
    sampler = key_to_sampler[namespace.sampler]

    return sampler.from_cli_arguments(problem, namespace)
