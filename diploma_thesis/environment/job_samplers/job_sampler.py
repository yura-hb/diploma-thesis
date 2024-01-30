
from abc import ABCMeta, abstractmethod

import torch

import environment
from environment import Problem


class JobSampler(metaclass=ABCMeta):

    def __init__(self, problem: Problem):
        self.problem = problem

    def number_of_jobs(self):
        return

    @abstractmethod
    def sample(self, job_id: int, moment: torch.FloatType) -> environment.Job:
        pass

    @abstractmethod
    def sample_next_arrival_time(self) -> torch.FloatTensor:
        pass
