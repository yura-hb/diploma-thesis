
from .simulator import Simulator
from .episodic import EpisodicSimulator
from .td import TDSimulator
from .configuration import RunConfiguration, EvaluateConfiguration
from .simulation import Simulation
from agents import Machine, WorkCenter
from typing import Dict
from tape import TapeModel

import logging
import simpy

key_to_class = {
    "episodic": EpisodicSimulator,
    "td": TDSimulator
}


def from_cli(machine: Machine,
             work_center: WorkCenter,
             tape: TapeModel,
             environment: simpy.Environment,
             logger: logging.Logger,
             parameters: Dict):
    cls = key_to_class[parameters['kind']]

    return cls(machine, work_center, tape, environment, logger)
