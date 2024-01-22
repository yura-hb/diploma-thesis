
import torch

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


