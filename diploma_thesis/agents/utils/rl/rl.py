
from abc import abstractmethod
from typing import List

import pandas as pd
import torch
import copy

from agents.base.agent import TrainingSample, Trajectory, Slice
from agents.utils.policy import Policy
from agents.utils.memory import Memory, Record
from agents.utils.nn import Loss, Optimizer
from agents.utils.return_estimator import ReturnEstimator
from utils import Loggable
from dataclasses import dataclass
from enum import StrEnum


class TrainSchedule(StrEnum):
    ON_TIMELINE = 'on_timeline'
    ON_STORE = 'on_store'

    @staticmethod
    def from_cli(parameters: dict):
        if 'train_schedule' not in parameters:
            return TrainSchedule.ON_TIMELINE

        return TrainSchedule(parameters['train_schedule'])


class RLTrainer(Loggable):

    def __init__(self,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator,
                 train_schedule: TrainSchedule = TrainSchedule.ON_TIMELINE):
        super().__init__()

        self.memory = memory
        self.loss = loss
        self.optimizer = optimizer
        self.return_estimator = return_estimator
        self._is_configured = False
        self.train_schedule = train_schedule
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

    def store(self, sample: TrainingSample):
        records = self.__prepare__(sample)
        records.info['episode'] = torch.full(records.reward.shape, sample.episode_id, device=records.reward.device)

        self.memory.store(records.view(-1))

        if self.train_schedule != TrainSchedule.ON_STORE:
            return

        self.__train__(sample.model)


    def clear(self):
        self.loss_cache = []
        self.memory.clear()

    @staticmethod
    def from_cli(parameters, memory: Memory, loss: Loss, optimizer: Optimizer, return_estimator: ReturnEstimator):
        pass

    def record_loss(self, loss: torch.Tensor, **kwargs):
        self.loss_cache += [dict(
            value=loss.detach().item(),
            optimizer_step=self.optimizer.step_count,
            lr=self.optimizer.learning_rate,
            **kwargs
        )]

    # TODO: - Test if records in sample aren't modified
    def __prepare__(self, sample: TrainingSample) -> Record:
        match sample:
            case Trajectory(_, records):
                updated = self.return_estimator.update_returns(records)
                updated = torch.cat(updated, dim=0)

                return updated
            case Slice(_, records):
                if len(records) == 1:
                    return records[0].view(-1)

                updated = self.return_estimator.update_returns(records)

                return updated[0]
            case _:
                raise ValueError(f'Unknown sample type: {type(sample)}')
