from typing import Dict

from tensordict.prototype import tensorclass

from .encoder import StateEncoder


class PlainEncoder(StateEncoder):

    @tensorclass
    class State:
        pass

    def encode(self, parameters: StateEncoder.Input) -> State:
        return self.State()

    @staticmethod
    def from_cli(parameters: Dict):
        return PlainEncoder()
