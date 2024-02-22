from functools import partial

from utils import from_cli
from .utils import Input as WorkCenterInput
from .work_center import WorkCenter
from .static import StaticWorkCenter
from .rl import RLWorkCenter

key_to_class = {
    "static": StaticWorkCenter,
    'rl': RLWorkCenter
}

from_cli = partial(from_cli, key_to_class=key_to_class)
