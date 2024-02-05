from dataclasses import dataclass
from typing import Dict


@dataclass
class Configuration:
    # The duration of the simulation in the number of jobs
    timespan: int = 2000
    # The number of machines per work center
    machines_per_work_center: int = 1
    # The number of work centers
    work_center_count: int = 1
    # Whether each work center on shopfloor will have a job in the beginning of simulation
    pre_assign_initial_jobs: bool = False
    # Breakdown ratio
    breakdown_ratio: float = 0.0

    @staticmethod
    def from_cli_arguments(configuration: Dict) -> 'Configuration':
        return Configuration(
            timespan=configuration['timespan'],
            machines_per_work_center=configuration['machines_per_work_center'],
            work_center_count=configuration['work_center_count'],
            pre_assign_initial_jobs=configuration.get('pre_assign_initial_jobs', False),
            breakdown_ratio=configuration.get('breakdown_ratio', 0.0)
        )
