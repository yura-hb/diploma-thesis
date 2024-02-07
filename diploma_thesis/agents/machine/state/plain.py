
from .encoder import StateEncoder
from dataclasses import dataclass


class PlainEncoder(StateEncoder):

    @dataclass
    class State:
        pass

    def encode(self, parameters: StateEncoder.Input) -> State:
        return self.State()
