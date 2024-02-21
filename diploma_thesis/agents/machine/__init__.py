from functools import partial

from utils import from_cli
from .dqn import DeepQAgent
from .machine import Machine
from .persisted import PersistedMachine
from .static import StaticMachine
from .utils import Input as MachineInput

key_to_class = {
    "static": StaticMachine,
    'dqn': DeepQAgent,
    'persisted': PersistedMachine,
}


from_cli = partial(from_cli, key_to_class=key_to_class)
