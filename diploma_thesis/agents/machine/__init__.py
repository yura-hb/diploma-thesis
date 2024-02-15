from .utils import Input as MachineInput
from .dqn import DeepQAgent
from .machine import Machine
from .static import StaticMachine
from .persisted import PersistedMachine

key_to_class = {
    "static": StaticMachine,
    'dqn': DeepQAgent,
    'persisted': PersistedMachine
}


def from_cli(parameters) -> Machine:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
