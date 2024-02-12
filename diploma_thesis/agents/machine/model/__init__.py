import logging

from .model import MachineModel
from .static import StaticModel as StaticMachineModel
from .rule import SchedulingRule

key_to_class = {
    "static": StaticMachineModel
}


def from_cli(parameters) -> MachineModel:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters['parameters'])

