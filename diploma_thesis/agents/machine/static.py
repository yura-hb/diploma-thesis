from typing import Dict

from .machine import Machine
from .model import MachineModel, from_cli as model_from_cli
from .state import StateEncoder, from_cli as state_encoder_from_cli


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
