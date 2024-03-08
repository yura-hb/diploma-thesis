
from abc import abstractmethod
from enum import StrEnum
from typing import List

import pandas as pd
import torch

from agents.base.agent import TrainingSample, Trajectory, Slice
from agents.base.state import GraphState
from agents.utils.memory import Memory, Record
from agents.utils.nn import Loss, Optimizer
from agents.utils.policy import Policy
from agents.utils.return_estimator import ReturnEstimator
from utils import Loggable

from torch_geometric.data import Batch

from .storage import *


class TrainSchedule(StrEnum):
    ON_TIMELINE = 'on_timeline'
    ON_STORE = 'on_store'
    ON_STORED_DATA_EXCLUSIVELY = 'on_stored_data_exclusively'

    @property
    def is_on_store(self):
        return self in [TrainSchedule.ON_STORE, TrainSchedule.ON_STORED_DATA_EXCLUSIVELY]

    @staticmethod
    def from_cli(parameters: dict):
        if 'train_schedule' not in parameters:
            return TrainSchedule.ON_TIMELINE

        return TrainSchedule(parameters['train_schedule'])


class RLTrainer(Loggable):

    def __init__(self,
                 is_episodic: bool,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator,
                 train_schedule: TrainSchedule = TrainSchedule.ON_TIMELINE):
        super().__init__()

        self.loss = loss
        self.optimizer = optimizer
        self._is_configured = False
        self.train_schedule = train_schedule

        self.storage = Storage(is_episodic, memory, return_estimator)
        self.loss_cache = []

    @abstractmethod
    def configure(self, model: Policy):
        self._is_configured = True

        if not self.optimizer.is_connected:
            self.optimizer.connect(model.parameters())

    def __train__(self, model: Policy):
        pass

    @property
    def is_configured(self):
        return self._is_configured

    def train_step(self, model: Policy):
        if self.train_schedule != TrainSchedule.ON_TIMELINE:
            return

        self.__train__(model)

    def loss_record(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_cache)

    def store(self, sample: TrainingSample, model: Policy):
        if self.train_schedule == TrainSchedule.ON_STORED_DATA_EXCLUSIVELY:
            self.storage.clear()

        self.storage.store(sample)

        if not self.train_schedule.is_on_store:
            return

        self.__train__(model)

    def clear(self):
        self.loss_cache = []
        self.storage.clear()

    def record_loss(self, loss: torch.Tensor, **kwargs):
        self.loss_cache += [dict(
            value=loss.detach().item(),
            optimizer_step=self.optimizer.step_count,
            lr=self.optimizer.learning_rate,
            **kwargs
        )]
