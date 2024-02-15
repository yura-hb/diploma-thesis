import logging

from .utils import Input as WorkCenterInput
from .work_center import WorkCenter
from .static import StaticWorkCenter
from .persisted import PersistedWorkCenter


key_to_class = {
    "static": StaticWorkCenter,
    'persisted': PersistedWorkCenter
}


def from_cli(parameters) -> WorkCenter:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
