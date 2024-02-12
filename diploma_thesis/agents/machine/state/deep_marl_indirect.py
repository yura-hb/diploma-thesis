from dataclasses import dataclass
from typing import Dict

from .encoder import StateEncoder


class DEEPMARLIndirectStateEncoder(StateEncoder):

    @dataclass
    class State:
        pass

    def encode(self, parameters: StateEncoder.Input) -> State:
        return self.State()

    @staticmethod
    def from_cli(parameters: Dict):
        return DEEPMARLIndirectStateEncoder()
