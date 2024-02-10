
from environment import Breakdown
from .no import No as NoBreakdown
from .dynamic import Dynamic as DynamicBreakdown

from typing import Dict

key_to_cls = {
    'no': NoBreakdown,
    'dynamic': DynamicBreakdown
}


def from_cli(parameters: Dict) -> Breakdown:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters.get('parameters', dict()))
