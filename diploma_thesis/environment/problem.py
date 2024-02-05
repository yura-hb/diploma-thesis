from dataclasses import dataclass
from typing import Dict


@dataclass
class Problem:
    # The duration of the simulation in the number of jobs
    timespan: int = 2000
    # The number of machines per work center
    machines_per_workcenter: int = 1
    # The number of work centers
    workcenter_count: int = 1

    @staticmethod
    def from_cli_arguments(configuration: Dict) -> 'Problem':
        return Problem(
            timespan=configuration['timespan'],
            machines_per_workcenter=configuration['machines_per_workcenter'],
            workcenter_count=configuration['workcenter_count']
        )

