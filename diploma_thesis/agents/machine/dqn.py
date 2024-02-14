

from .machine import *
from typing import Dict


class DeepQAgent(Machine):

    def is_trainable(self):
        return super().is_trainable()

    def train_step(self):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])
        memory = memory_from_cli(parameters['memory'])

        return DeepQAgent(model=model, state_encoder=encoder, memory=memory)
