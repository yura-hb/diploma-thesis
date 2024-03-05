
from agents.base.state import GraphState

from tensordict.prototype import tensorclass
from .encoder import *


class AuxiliaryGraphEncoder(StateEncoder):

    @tensorclass
    class State(GraphState):
        pass

    def encode(self, parameters: StateEncoder.Input) -> State:
        if parameters.graph is None:
            raise ValueError("Graph is not provided")

        a = 10

