from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
import torch.cuda

from agents.utils.nn import Loss, Optimizer
from agents.utils.policy import Policy
from utils import Loggable
from .storage import *


@dataclass
class TrainSchedule:
    class Kind(StrEnum):
        ON_TIMELINE = 'on_timeline'
        ON_STORE = 'on_store'
        ON_STORED_DATA_EXCLUSIVELY = 'on_stored_data_exclusively'

    kind: Kind
    train_every: int = 1

    @property
    def is_on_store(self):
        return self.kind in [TrainSchedule.Kind.ON_STORE, TrainSchedule.Kind.ON_STORED_DATA_EXCLUSIVELY]

    @staticmethod
    def from_cli(parameters: dict):
        if 'train_schedule' not in parameters:
            return TrainSchedule(TrainSchedule.Kind.ON_TIMELINE)

        return TrainSchedule(parameters['train_schedule'], train_every=parameters.get('train_every', 1))


class RLTrainer(Loggable):

    def __init__(self,
                 is_episodic: bool,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator,
                 train_schedule: TrainSchedule = TrainSchedule(TrainSchedule.Kind.ON_TIMELINE),
                 device: str = 'cpu'):
        super().__init__()

        self.loss = loss
        self.optimizer = optimizer
        self.train_schedule = train_schedule
        self.return_estimator = return_estimator
        self.device = device

        self.counter = 0

        self._is_configured = False
        self.storage = Storage(is_episodic, memory, return_estimator)
        self.loss_cache = []

    def configure(self, model: Policy):
        self._is_configured = True

        if not self.optimizer.is_connected:
            self.optimizer.connect(model.parameters())

    @property
    def is_configured(self):
        return self._is_configured

    def train_step(self, model: Policy):
        if not self.is_configured:
            return

        if self.train_schedule.is_on_store:
            return

        self.__train_step__(model)

    def __train__(self, model: Policy):
        pass

    def loss_record(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_cache)

    def store(self, sample: TrainingSample, model: Policy):
        if self.train_schedule.kind == TrainSchedule.Kind.ON_STORED_DATA_EXCLUSIVELY:
            self.storage.clear()

        self.storage.store(sample)

        if not self.train_schedule.is_on_store:
            return

        self.counter += 1

        if self.counter % self.train_schedule.train_every == 0:
            self.counter = 0
            self.__train_step__(model)

            if self.train_schedule.kind == TrainSchedule.Kind.ON_STORED_DATA_EXCLUSIVELY:
                self.storage.clear()

    def __train_step__(self, model: Policy):
        import time

        start = time.time()

        self.__train__(model)

        torch.cuda.empty_cache()

        print(f'Train step: { time.time() - start } Optimizer Step: { self.optimizer.step_count }')

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

    @staticmethod
    def step(compute_loss, optimizer):
        optimizer.zero_grad()

        def compute_grad():
            loss, output = compute_loss()
            loss.backward()

            return loss, output

        return optimizer.step(compute_grad)
