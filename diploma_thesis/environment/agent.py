
from abc import ABCMeta, abstractmethod
from environment import Job, Machine, WorkCenter, Machine
from dataclasses import dataclass
from typing import List


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
    def schedule(self, shop_floor_id: int, machine: Machine, now: int) -> Job | WaitInfo:
        pass

    @abstractmethod
    def route(self, shop_floor_id: int, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        pass

    @abstractmethod
    def will_produce(self, shop_floor_id: int, job: Job, machine: Machine):
        """
        Will be triggered before the production of job on machine
        """
        ...

    @abstractmethod
    def did_produce(self, shop_floor_id: int, job: Job, machine: Machine):
        """
        Will be triggered after the production of job on machine
        """
        ...

    @abstractmethod
    def will_dispatch(self, shop_floor_id: int, job: Job, work_center: WorkCenter):
        """
        Will be triggered before dispatch of job on the work-center
        """
        ...

    @abstractmethod
    def did_dispatch(self, shop_floor_id: int, job: Job, work_center: WorkCenter, machine: Machine):
        """
        Will be triggered after the dispatch of job to the machine
        """
        ...

    @abstractmethod
    def did_finish_dispatch(self, shop_floor_id: int, work_center: WorkCenter):
        """
        Will be triggered after the dispatch of job on the work-center
        """
        ...

    @abstractmethod
    def did_complete(self, shop_floor_id: int, job: Job):
        """
        Will be triggered after the completion of job
        """
        ...
