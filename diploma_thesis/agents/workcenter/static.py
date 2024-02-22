from typing import Dict

from .model import StaticWorkCenterModel
from .work_center import *


class StaticWorkCenter(WorkCenter):

    def __init__(self, model: StaticWorkCenterModel, state_encoder: StateEncoder):
        super().__init__(model=model, state_encoder=state_encoder)

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])

        return StaticWorkCenter(model=model, state_encoder=encoder)
