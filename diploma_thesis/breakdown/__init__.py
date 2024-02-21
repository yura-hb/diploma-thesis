from functools import partial

from utils import from_cli
from .dynamic import Dynamic as DynamicBreakdown
from .no import No as NoBreakdown

key_to_cls = {
    'no': NoBreakdown,
    'dynamic': DynamicBreakdown
}

from_cli = partial(from_cli, key_to_class=key_to_cls)

