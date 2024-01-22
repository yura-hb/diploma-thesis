
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from problem.job import Job
from problem.machine import Machine
from typing import List


@dataclass
class WorkCenterState:
    work_center_idx: int

    machines: List[Machine]


class RoutingRule(metaclass=ABCMeta):

    @abstractmethod
    def select_machine(self, job: Job, state: WorkCenterState) -> Machine:
        ...
