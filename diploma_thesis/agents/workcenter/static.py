import logging

from .work_center import WorkCenter
from .model import StaticWorkCenterModel, from_cli as model_from_cli
from .state import StateEncoder, from_cli as state_encoder_from_cli
from typing import Dict


class StaticWorkCenter(WorkCenter):

    def __init__(self, model: StaticWorkCenterModel, state_encoder: StateEncoder):
        super().__init__(model=model, state_encoder=state_encoder, memory=None)

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass

    def state_dict(self):
        return {
            'model': self.model,
            'encoder': self.state_encoder
        }

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])

        return StaticWorkCenter(model=model, state_encoder=encoder)

    @classmethod
    def load_from_parameters(cls, parameters):
        model = parameters['model']
        encoder = parameters['encoder']

        return StaticWorkCenter(
            model=model,
            state_encoder=encoder
        )
