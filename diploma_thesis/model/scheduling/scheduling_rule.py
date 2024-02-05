
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import environment
from environment.job import Job


@dataclass
class WaitInfo:
    """
    Information about scheduling decision
    """
    wait_time: int = 0


class SchedulingRule(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, machine: environment.Machine, now: int) -> Job | WaitInfo:
        pass
