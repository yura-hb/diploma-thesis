from functools import partial

from utils import from_cli

from .utils import Input as MachineInput
from .static import StaticMachine
from .rl import RLMachine
from .marl import MARLMachine


key_to_class = {
    "static": StaticMachine,
    'rl': RLMachine,
    'marl': MARLMachine
}


from_cli = partial(from_cli, key_to_class=key_to_class)
