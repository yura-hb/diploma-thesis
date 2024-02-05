
from abc import ABCMeta, abstractmethod

import simpy
import torch

import environment
from environment import Problem


class JobSampler(metaclass=ABCMeta):

    def __init__(self, problem: Problem, environment: simpy.Environment):
        self.problem = problem
        self.environment = environment

    def number_of_jobs(self):
        return

    @abstractmethod
    def sample(self, job_id: int, initial_work_center_idx: int, moment: torch.FloatType) -> environment.Job:
        pass

    @abstractmethod
    def sample_next_arrival_time(self) -> torch.FloatTensor:
        pass
