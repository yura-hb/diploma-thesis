
from abc import ABCMeta, abstractmethod
from environment import ShopFloor
from dispatch.job_sampler import JobSampler

from utils import Loggable


class InitialJobAssignment(Loggable, metaclass=ABCMeta):

    @abstractmethod
    def make_jobs(self, shop_floor: ShopFloor, sampler: JobSampler):
        pass
