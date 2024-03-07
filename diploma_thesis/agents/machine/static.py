from typing import Dict

from agents.base.agent import Agent, Key, TrainingSample
from agents.utils.memory import Record
from environment import MachineKey
from .model import MachineModel, from_cli as model_from_cli
from .state import StateEncoder, from_cli as state_encoder_from_cli


class StaticMachine(Agent[MachineKey]):

    def __init__(self, model: MachineModel, state_encoder: StateEncoder):
        super().__init__(model=model, state_encoder=state_encoder)

    @property
    def is_trainable(self):
        return False

    @property
    def is_distributed(self):
        return False

    def train_step(self):
        pass

    def store(self, key: Key, sample: TrainingSample):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])

        return StaticMachine(model=model, state_encoder=encoder)
