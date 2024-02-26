
from dataclasses import dataclass
from environment import Machine
from agents.base import Graph


@dataclass
class Input:
    machine: Machine
    now: float
    graph: Graph | None
