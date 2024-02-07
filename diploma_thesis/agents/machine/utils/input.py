
from dataclasses import dataclass
from environment import Machine


@dataclass
class Input:
    machine: Machine
    now: float
