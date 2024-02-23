from functools import partial

from utils import from_cli
from .rl import RLWorkCenter
from .static import StaticWorkCenter
from .utils import Input as WorkCenterInput

key_to_class = {
    "static": StaticWorkCenter,
    'rl': RLWorkCenter
}

from_cli = partial(from_cli, key_to_class=key_to_class)
