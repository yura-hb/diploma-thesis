
from .initial_job_assignment import *
from typing import Dict
import torch


class JSPStatic(InitialJobAssignment):

    def __init__(self, path: str):
        self.path = path

    def make_jobs(self, shop_floor: ShopFloor, sampler: JobSampler):
        with open(self.path, 'r') as f:
            lines = f.readlines()

        job_count, machines_count = lines[0].split()
        job_count, machines_count = int(job_count), int(machines_count)

        assert machines_count == shop_floor.configuration.configuration.work_center_count and \
               shop_floor.configuration.configuration.machines_per_work_center == 1, \
               "Shop floor must have the same configration as training instance"

        for index, line in enumerate(lines[1:]):
            records = line.split()
            records = list(map(int, records))
            step_idx = records[::2]
            processing_times = records[1::2]

            job = sampler.sample(index, step_idx[0], 0)

            job.step_idx = torch.tensor(step_idx, dtype=torch.int32)
            job.processing_times = torch.tensor(processing_times, dtype=torch.float32)
            job.processing_times = torch.atleast_2d(job.processing_times).T

            job.with_due_at(job.processing_times.sum() * 1.5)

            yield job

    @staticmethod
    def from_cli(parameters: Dict):
        return JSPStatic(path=parameters['path'])

