
from abc import abstractmethod

import pandas as pd
import torch

from agents.base.model import NNModel
from agents.utils.memory import Record, Memory, NotReadyException
from agents.utils.nn import LossCLI, OptimizerCLI

from utils import Loggable


class RLTrainer(Loggable):

    def __init__(self, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI):
        super().__init__()

        self.memory = memory
        self.loss = loss
        self.optimizer = optimizer
        self._is_configured = False
        self.loss_cache = []

    @abstractmethod
    def configure(self, model: NNModel):
        self._is_configured = True

        if not self.optimizer.is_connected:
            self.optimizer.connect(model.parameters())

    @property
    def is_configured(self):
        return self._is_configured

    @abstractmethod
    def train_step(self, model: NNModel):
        pass

    def loss_record(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_cache)

    def store(self, record: Record):
        self.memory.store(record.view(-1))

    def clear(self):
        self.loss_cache = []
        self.memory.clear()

    @staticmethod
    def from_cli(parameters, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI):
        pass

    def record_loss(self, loss: torch.FloatTensor, **kwargs):
        self.loss_cache += [dict(
            value=loss.detach().item(),
            optimizer_step=self.optimizer.step_count,
            lr=self.optimizer.learning_rate,
            **kwargs
        )]
