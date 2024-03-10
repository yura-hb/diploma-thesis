
from .policy import Policy, Record as PolicyRecord
from .discrete_action import DiscreteAction
from .flexible_action import FlexibleAction

from utils import from_cli
from functools import partial

key_to_cls = {
    "discrete_action": DiscreteAction,
    "flexible_action": FlexibleAction
}

from_cli = partial(from_cli, key_to_class=key_to_cls)
