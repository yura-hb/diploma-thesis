
from dataclasses import dataclass

from agents.base import Graph
from environment import Machine


@dataclass
class Input:
    machine: Machine
    now: float
    graph: Graph | None
