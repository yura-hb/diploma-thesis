
from .transition import GraphTransition

from utils import from_cli

from .no import No

from functools import partial

key_to_class = {
    'no': No,
    'base': GraphTransition
}

from_cli = partial(from_cli, key_to_class=key_to_class)
