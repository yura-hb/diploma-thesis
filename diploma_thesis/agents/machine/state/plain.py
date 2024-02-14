from dataclasses import dataclass
from typing import Dict

from .encoder import StateEncoder
from tensordict.prototype import tensorclass


class PlainEncoder(StateEncoder):

    @tensorclass
    class State:
        pass

    def encode(self, parameters: StateEncoder.Input) -> State:
        return self.State()

    @staticmethod
    def from_cli(parameters: Dict):
        return PlainEncoder()
