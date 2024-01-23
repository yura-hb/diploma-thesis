
import torch
import argparse

from dataclasses import dataclass
from typing import Tuple

# TODO: Think, how to ingest the number of machines


@dataclass
class Problem:
    # The duration of the simulation in the number of jobs
    timespan: int = 1000
    # Range of processing times [min, max]
    processing_time: Tuple[int, int] = (1, 10)
    # Tightness factor
    tightness_factor: float = 1.0
    # Expected utilization rate of machines
    expected_utilization: float = 0.75
    # The number of machines per work center
    machines_per_workcenter: int = 1
    # The number of work centers
    workcenter_count: int = 1
    # Seed
    seed: int = 0

    @property
    def beta(self) -> torch.Tensor:
        """
        Returns: Defines the expected number of jobs per time unit to meet the expected utilization for shopfloor
        """
        return self.expected_processing_time / (self.machines_per_workcenter * self.expected_utilization)

    @property
    def job_count(self) -> torch.Tensor:
        return torch.round(self.timespan / self.beta).to(torch.long)

    @property
    def expected_processing_time(self) -> torch.Tensor:
        distance = self.processing_time[1] - self.processing_time[0]

        return torch.Tensor([self.processing_time[0] + distance / 2])

    def sample_processing_times(self, count: int, generator: torch.Generator) -> torch.Tensor:
        return torch.randint(
            low=self.processing_time[0],
            high=self.processing_time[1],
            size=(count,),
            generator=generator
        )

    def sample_next_arrival_time(self, count: int) -> torch.Tensor:
        exponential = torch.distributions.Exponential(self.beta)

        return exponential.rsample((count,))

    @staticmethod
    def add_cli_arguments(parser: argparse.ArgumentParser):
        sub_parsers = parser.add_subparsers(dest="problem")
        sub_parser = sub_parsers.add_parser(name="configuration")

        sub_parser.add_argument(
            "--timespan",
            help="The duration of the simulation",
            type=int,
            default=1000
        )

        sub_parser.add_argument(
            "--processing-time",
            help="Parameters of uniform distribution generating processing times of job [min, max]",
            nargs=2,
            type=int,
            default=[1, 10]
        )

        sub_parser.add_argument(
            "--tightness-factor",
            help="Tightness factor",
            type=float,
            default=2.0
        )

        sub_parser.add_argument(
            "--expected-utilization",
            help="Expected utilization rate of machines",
            type=float,
            default=0.75
        )

        sub_parser.add_argument(
            "--machines-per-workcenter",
            help="The number of machines per work center",
            type=int,
            default=1
        )

        sub_parser.add_argument(
            "--workcenter-count",
            help="The number of work centers",
            type=int,
            default=1
        )

        sub_parser.add_argument(
            "--seed",
            help="Seed",
            type=int,
            default=0
        )

    @staticmethod
    def from_cli_arguments(namespace: argparse.Namespace) -> 'Problem':
        print(namespace)

        return Problem(
            timespan=namespace.timespan,
            processing_time=tuple(namespace.processing_time),
            tightness_factor=namespace.tightness_factor,
            expected_utilization=namespace.expected_utilization,
            machines_per_workcenter=namespace.machines_per_workcenter,
            workcenter_count=namespace.workcenter_count,
            seed=namespace.seed
        )

