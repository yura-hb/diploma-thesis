import argparse

from dataclasses import dataclass


@dataclass
class Problem:
    # The duration of the simulation in the number of jobs
    timespan: int = 2000
    # The number of machines per work center
    machines_per_workcenter: int = 1
    # The number of work centers
    workcenter_count: int = 1

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

        return sub_parser

    @staticmethod
    def from_cli_arguments(namespace: argparse.Namespace) -> 'Problem':
        return Problem(
            timespan=namespace.timespan,
            machines_per_workcenter=namespace.machines_per_workcenter,
            workcenter_count=namespace.workcenter_count
        )

