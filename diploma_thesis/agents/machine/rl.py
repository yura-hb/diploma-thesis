from typing import Dict

from agents.utils import TrainingPhase
from agents.utils.memory import Record
from agents.utils.rl import RLTrainer, from_cli as rl_trainer_from_cli
from utils import filter
from .machine import *


class RLMachine(Machine):

    def __init__(self, model: NNMachineModel, state_encoder: StateEncoder, trainer: RLTrainer):
        super().__init__(model, state_encoder)

        self.trainer = trainer

    @property
    def is_trainable(self):
        return True

    @filter(lambda self: self.phase == TrainingPhase())
    def train_step(self):
        self.trainer.train_step(self.model)

    @filter(lambda self, *args: self.phase == TrainingPhase())
    def store(self, key: MachineKey, record: Record):
        self.trainer.store(record)

    def schedule(self, parameters):
        result = super().schedule(parameters)

        if not self.trainer.is_configured:
            self.trainer.configure(self.model)

        return result

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])
        trainer = rl_trainer_from_cli(parameters['trainer'])

        assert isinstance(model, NNMachineModel), f"Model must conform to NNModel"

        return RLMachine(model, encoder, trainer)
