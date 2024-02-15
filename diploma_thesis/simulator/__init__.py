
from typing import Dict

from agents import Machine, WorkCenter
from tape import TapeModel
from .configuration import RunConfiguration, EvaluateConfiguration
from .episodic import EpisodicSimulator
from .simulation import Simulation
from .simulator import Simulator
from .td import TDSimulator

key_to_class = {
    "episodic": EpisodicSimulator,
    "td": TDSimulator
}


def from_cli(machine: Machine, work_center: WorkCenter, tape: TapeModel, parameters: Dict):
    cls = key_to_class[parameters['kind']]

    return cls(machine, work_center, tape)
