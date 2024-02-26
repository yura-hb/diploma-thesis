from typing import Dict

import torch
from tensordict.prototype import tensorclass

from agents.base.state import GraphState
from environment import JobReductionStrategy, Job, Machine
from .encoder import StateEncoder
from ...base.encoder import Input


class GraphEncoder(StateEncoder):

    @tensorclass
    class State(GraphState):
        pass

    def __init__(self):
        super().__init__()

    def encode(self, parameters: StateEncoder.Input) -> State:
        shop_floor = parameters.machine

