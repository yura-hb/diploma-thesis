

from .initial_job_assignment import *
from typing import Dict


class No(InitialJobAssignment):

    def make_jobs(self, shop_floor: ShopFloor, sampler: JobSampler):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        return No()
