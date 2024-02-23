from typing import Dict

from .machine import *
from ..base.agent import Key
from ..utils.memory import Record


class StaticMachine(Machine):

    def __init__(self, model: MachineModel, state_encoder: StateEncoder):
        super().__init__(model=model, state_encoder=state_encoder)

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass

    def store(self, key: Key, record: Record):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])

        return StaticMachine(model=model, state_encoder=encoder)
