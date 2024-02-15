
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import environment
from environment import Job, Context


@dataclass
class WaitInfo:
    """
    Information about scheduling decision
    """
    wait_time: int = 0


class Agent(metaclass=ABCMeta):
    """
    Support Class to handle the events in the shop-floor
    """

    @abstractmethod
    def schedule(self, context: Context, machine: 'environment.Machine') -> Job | WaitInfo:
        pass

    @abstractmethod
    def route(self, context: Context, work_center: 'environment.WorkCenter', job: Job) -> 'Machine | None':
        pass
