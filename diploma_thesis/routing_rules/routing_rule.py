
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

from environment.job import Job


@dataclass
class WorkCenterState:
    work_center_idx: int

    machines: List['Machine']


class RoutingRule(metaclass=ABCMeta):
    @abstractmethod
    def select_machine(self, job: Job, state: WorkCenterState) -> 'Machine':
        ...
