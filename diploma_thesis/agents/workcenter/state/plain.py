import logging

from .encoder import StateEncoder
from dataclasses import dataclass
from typing import Dict


class PlainEncoder(StateEncoder):

    @dataclass
    class State:
        pass

    def encode(self, parameters: StateEncoder.Input) -> State:
        return self.State()

    @staticmethod
    def from_cli(parameters: Dict):
        return PlainEncoder()
