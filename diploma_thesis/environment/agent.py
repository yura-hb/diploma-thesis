
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import environment
from environment import Job, Context


class Agent(metaclass=ABCMeta):
    """
    Support Class to handle the events in the shop-floor
    """

    @abstractmethod
    def schedule(self, context: Context, machine: 'environment.Machine') -> Job | None:
        pass

    @abstractmethod
    def route(self, context: Context, job: Job, work_center: 'environment.WorkCenter') -> 'Machine | None':
        pass
