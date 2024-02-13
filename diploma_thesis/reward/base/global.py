
from abc import ABCMeta, abstractmethod
from .reward_model import RewardModel
from environment import ShopFloor, Job


class Global(RewardModel, metaclass=ABCMeta):
    """
    Global
    """

    @abstractmethod
    def reward(self, job: Job, shop_floor: ShopFloor) -> float:
        pass

