
from abc import abstractmethod

from agents.base.model import NNModel
from agents.utils.memory import Record, Memory
from agents.utils.nn import LossCLI, OptimizerCLI


class RLTrainer:

    def __init__(self, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI):
        self.memory = memory
        self.loss = loss
        self.optimizer = optimizer
        self._is_configured = False

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

    def store(self, record: Record):
        self.memory.store(record.view(-1))

    @staticmethod
    def from_cli(parameters, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI):
        pass
