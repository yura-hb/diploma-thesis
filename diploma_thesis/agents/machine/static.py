from typing import Dict

from .machine import *


class StaticMachine(Machine):

    def __init__(self, model: MachineModel, state_encoder: StateEncoder):
        super().__init__(model=model, state_encoder=state_encoder, memory=None)

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])

        return StaticMachine(model=model, state_encoder=encoder)