
from abc import abstractmethod
from enum import StrEnum

import pandas as pd

from agents.utils.run_configuration import RunConfiguration
from agents.utils.nn import Loss, Optimizer
from agents.utils.policy import Policy
from utils import Loggable
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
        self.train_schedule = train_schedule
        self.return_estimator = return_estimator
        self.run_configuration: RunConfiguration = None

        self._is_configured = False
        self.storage = Storage(is_episodic, memory, return_estimator)
        self.loss_cache = []

    def configure(self, model: Policy, configuration: RunConfiguration):
        self._is_configured = True

        self.run_configuration = configuration

        if not self.optimizer.is_connected:
            self.optimizer.connect(model.parameters())

    @property
    def is_configured(self):
        return self._is_configured

    def train_step(self, model: Policy):
        if not self.is_configured:
            return

        if self.train_schedule != TrainSchedule.ON_TIMELINE:
            return

        self.__train__(model)

    def __train__(self, model: Policy):
        pass

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
            value=loss.detach().cpu().item(),
            optimizer_step=self.optimizer.step_count,
            lr=self.optimizer.learning_rate,
            **kwargs
        )]

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
