
from .simulator import Simulator
from .episodic import EpisodicSimulator
from .td import TDSimulator
from .configuration import RunConfiguration, EvaluateConfiguration
from agents import Machine, WorkCenter
from typing import Dict

key_to_class = {
    "episodic": EpisodicSimulator,
    "td": TDSimulator
}


def from_cli(machine: Machine, work_center: WorkCenter, reward_model, logger, parameters: Dict):
    cls = key_to_class[parameters['kind']]

    return cls(work_center, machine, reward_model, logger)
