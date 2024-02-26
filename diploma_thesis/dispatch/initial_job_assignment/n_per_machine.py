
from .initial_job_assignment import *
from typing import Dict


class NPerMachine(InitialJobAssignment):

    def __init__(self, n: int = 1):
        self.n = n

    def make_jobs(self, shop_floor: ShopFloor, sampler: JobSampler):
        for machine in shop_floor.machines:
            job = sampler.sample(
                job_id=shop_floor.new_job_id,
                initial_work_center_idx=machine.key.work_center_id,
                moment=shop_floor.now
            )

            yield job

    @staticmethod
    def from_cli(parameters: Dict):
        return NPerMachine(n=parameters['n'])
