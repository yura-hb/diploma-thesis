from .dqn import DeepQAgent
from .machine import Machine
from .static import StaticMachine
from .utils import Input as MachineInput

key_to_class = {
    "static": StaticMachine,
    'dqn': DeepQAgent
}


def from_cli(parameters) -> Machine:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
