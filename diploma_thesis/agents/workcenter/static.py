from typing import Dict

from agents.base.agent import Agent
from agents.utils.memory import Record
from environment import WorkCenterKey
from .model import WorkCenterModel, from_cli as model_from_cli
from .state import StateEncoder, from_cli as state_encoder_from_cli


class StaticWorkCenter(Agent[WorkCenterKey]):

    def __init__(self, model: WorkCenterModel, state_encoder: StateEncoder):
        super().__init__(model=model, state_encoder=state_encoder)

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass

    def store(self, key: WorkCenterKey, record: Record):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])

        return StaticWorkCenter(model=model, state_encoder=encoder)
