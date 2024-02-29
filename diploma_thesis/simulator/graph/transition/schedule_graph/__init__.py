
from .transition import ScheduleTransition
from .complete import CompleteTransition
from .compressed import CompressedTransition
from utils import from_cli
from functools import partial

key_to_class = {
    'complete': CompleteTransition,
    'compressed': CompressedTransition
}

from_cli = partial(from_cli, key_to_class=key_to_class)


