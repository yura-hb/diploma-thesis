
from abc import ABCMeta, abstractmethod

import torch

import environment
from environment import Configuration
from utils import Loggable


class JobSampler(Loggable, metaclass=ABCMeta):

    def __init__(self, problem: Configuration):
        self.problem = problem

        super().__init__()

    def connect(self, generator: torch.Generator):
        pass

    @abstractmethod
    def number_of_jobs(self):
        return

    @abstractmethod
    def sample(self, job_id: int, initial_work_center_idx: int, moment: torch.FloatType) -> environment.Job:
        pass

    @abstractmethod
    def sample_next_arrival_time(self) -> torch.FloatTensor:
        pass
