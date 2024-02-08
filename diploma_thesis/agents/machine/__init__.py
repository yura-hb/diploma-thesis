
from .utils import Input as MachineInput
from .machine import Machine
from .static import StaticMachine


key_to_class = {
    "static": StaticMachine
}


def from_cli(parameters) -> Machine:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
