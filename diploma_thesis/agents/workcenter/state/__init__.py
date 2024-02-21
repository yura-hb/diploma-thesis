from .encoder import StateEncoder
from .plain import PlainEncoder
from utils import from_cli
from functools import partial

key_to_class = {
    "plain": PlainEncoder
}

from_cli = partial(from_cli, key_to_class=key_to_class)
