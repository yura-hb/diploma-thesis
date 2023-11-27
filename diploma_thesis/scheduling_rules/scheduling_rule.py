
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import List
from problem.job import Job


@dataclass
class MachineState:
    """
    Structure to represent machine state as well as support metrics to perform scheduling
    """
    queue: List[Job] = field(default_factory=list)
    # The time moment scheduling is performed
    now: int = 0


@dataclass
class WaitInfo:
    """
    Information about scheduling decision
    """
    wait_time: int = 0


class SchedulingRule(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        pass
