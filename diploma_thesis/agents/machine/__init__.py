from functools import partial

from utils import from_cli

from .utils import Input as MachineInput
from .machine import Machine
from .static import StaticMachine
from .rl import RLMachine


key_to_class = {
    "static": StaticMachine,
    'rl': RLMachine,
}


from_cli = partial(from_cli, key_to_class=key_to_class)
