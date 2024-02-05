
from abc import ABCMeta, abstractmethod
from typing import List

from environment import Machine
from environment.job import Job


class RoutingRule(metaclass=ABCMeta):
    @abstractmethod
    def select_machine(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        ...
