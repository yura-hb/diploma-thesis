
from abc import abstractmethod
from typing import List

import pandas as pd
import torch

from agents.utils.policy import Policy
from agents.utils.memory import Record, Memory
from agents.utils.nn import LossCLI, OptimizerCLI
from agents.utils.return_estimator import ReturnEstimator
from utils import Loggable


class RLTrainer(Loggable):

    def __init__(self, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI, return_estimator: ReturnEstimator):
        super().__init__()

        self.memory = memory
        self.loss = loss
        self.optimizer = optimizer
        self.return_estimator = return_estimator
        self._is_configured = False
        self.loss_cache = []

    @abstractmethod
    def configure(self, model: Policy):
        self._is_configured = True

        if not self.optimizer.is_connected:
            self.optimizer.connect(model.parameters())

    @property
    def is_configured(self):
        return self._is_configured

    @abstractmethod
    def train_step(self, model: Policy):
        pass

    def loss_record(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_cache)

    def store(self, record: Record | List[Record]):
        self.memory.store(record.view(-1))

    def clear(self):
        self.loss_cache = []
        self.memory.clear()

    @staticmethod
    def from_cli(parameters, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI, return_estimator: ReturnEstimator):
        pass

    def record_loss(self, loss: torch.FloatTensor, **kwargs):
        self.loss_cache += [dict(
            value=loss.detach().item(),
            optimizer_step=self.optimizer.step_count,
            lr=self.optimizer.learning_rate,
            **kwargs
        )]
