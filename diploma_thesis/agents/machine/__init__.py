from functools import partial

from utils import from_cli

from .utils import Input as MachineInput
from .machine import Machine
from .static import StaticMachine
from .dqn import DeepQAgent


key_to_class = {
    "static": StaticMachine,
    'dqn': DeepQAgent,
}


from_cli = partial(from_cli, key_to_class=key_to_class)
