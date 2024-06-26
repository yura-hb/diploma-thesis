from dataclasses import dataclass
from typing import Dict


@dataclass
class Configuration:
    # The duration of the run in the number of jobs
    timespan: int = 2000
    # The number of machines per work center
    machines_per_work_center: int = 1
    # The number of work centers
    work_center_count: int = 1
    # Whether machine should automatically perform naive action
    deduce_naive_actions: int = False

    @staticmethod
    def from_cli_arguments(configuration: Dict) -> 'Configuration':
        return Configuration(
            timespan=configuration['timespan'],
            machines_per_work_center=configuration['machines_per_work_center'],
            work_center_count=configuration['work_center_count'],
            deduce_naive_actions=configuration.get('deduce_naive_actions', False)
        )
