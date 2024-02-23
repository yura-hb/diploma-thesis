from agents.utils import TrainingPhase
from agents.utils.rl import RLTrainer
from utils import filter
from .agent import *
from .model import NNModel


class RLAgent(Generic[Key], Agent[Key]):

    def __init__(self, model: NNModel, state_encoder: StateEncoder, trainer: RLTrainer):
        super().__init__(model, state_encoder)

        self.trainer = trainer

    @property
    def is_trainable(self):
        return True

    @filter(lambda self: self.phase == TrainingPhase())
    def train_step(self):
        self.trainer.train_step(self.model)

    @filter(lambda self, *args: self.phase == TrainingPhase())
    def store(self, key: Key, record: Record):
        self.trainer.store(record)

    def loss_record(self):
        return self.trainer.loss_record()

    def clear_memory(self):
        self.trainer.clear()

    def schedule(self, parameters):
        result = super().schedule(parameters)

        if not self.trainer.is_configured:
            self.trainer.configure(self.model)

        return result
