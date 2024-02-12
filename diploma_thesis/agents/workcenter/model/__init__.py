import logging

from .model import WorkCenterModel
from .static import StaticModel as StaticWorkCenterModel
from .rule import RoutingRule

key_to_class = {
    "static": StaticWorkCenterModel
}


def from_cli(parameters) -> WorkCenterModel:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
