
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

import environment

from environment import Job, Machine


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
    def schedule(self, shop_floor: 'environment.ShopFloor', machine: Machine, now: int) -> Job | WaitInfo:
        pass

    @abstractmethod
    def route(self,
              shop_floor: 'environment.ShopFloor',
              job: Job,
              work_center_idx: int,
              machines: List[Machine]) -> 'Machine | None':
        pass
