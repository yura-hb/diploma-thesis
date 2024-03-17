from typing import Dict

from .encoder import *


class PlainEncoder(StateEncoder):

    def encode(self, parameters: StateEncoder.Input) -> State:
        return State(batch_size=[])

    @staticmethod
    def from_cli(parameters: Dict):
        return PlainEncoder()
