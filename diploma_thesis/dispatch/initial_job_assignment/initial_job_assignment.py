
from abc import ABCMeta, abstractmethod
from environment import ShopFloor
from dispatch.job_sampler import JobSampler


class InitialJobAssignment(metaclass=ABCMeta):

    @abstractmethod
    def make_jobs(self, shop_floor: ShopFloor, sampler: JobSampler):
        pass
