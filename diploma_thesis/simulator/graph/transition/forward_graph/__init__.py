
from .transition import ForwardTransition
from .adding_arc import AddingArcTransition
from .complete import CompleteTransition
from .compressed import CompressedTransition

from utils import from_cli
from functools import partial

key_to_class = {
    'adding_arc': AddingArcTransition,
    'complete': CompleteTransition,
    'compressed': CompressedTransition
}

from_cli = partial(from_cli, key_to_class=key_to_class)
