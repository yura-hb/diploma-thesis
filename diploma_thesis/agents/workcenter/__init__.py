from functools import partial

from utils import from_cli
from .utils import Input as WorkCenterInput
from .work_center import WorkCenter
from .static import Static
from .rl import RLAgent

key_to_class = {
    "static": Static,
    'rl': RLAgent
}

from_cli = partial(from_cli, key_to_class=key_to_class)
