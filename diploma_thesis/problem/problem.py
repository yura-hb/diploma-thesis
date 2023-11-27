
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

    @property
    def beta(self) -> torch.Tensor:
        return torch.mean(self.processing_time) / self.expected_utilization

    @property
    def job_count(self) -> torch.Tensor:
        return torch.round(self.timespan / self.beta).to(torch.long)

    @property
    def expected_processing_time(self) -> torch.Tensor:
        return torch.mean(self.processing_time)

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
