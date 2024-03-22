
from dataclasses import dataclass

from agents.base import Graph
from environment import Machine

from tensordict import TensorDict


@dataclass
class Input:
    machine: Machine
    now: float
    graph: Graph | None
    memory: TensorDict | None
