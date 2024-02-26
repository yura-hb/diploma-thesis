
from .initial_job_assignment import InitialJobAssignment
from utils import from_cli
from functools import partial

from .no import No
from .n_per_machine import NPerMachine

key_to_class = {
    'no': No,
    'n_per_machine': NPerMachine
}


from_cli = partial(from_cli, key_to_class=key_to_class)
