from functools import partial

from utils import from_cli
from .persisted import PersistedWorkCenter
from .static import StaticWorkCenter
from .utils import Input as WorkCenterInput
from .work_center import WorkCenter

key_to_class = {
    "static": StaticWorkCenter,
    'persisted': PersistedWorkCenter
}

from_cli = partial(from_cli, key_to_class=key_to_class)
