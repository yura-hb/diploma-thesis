from functools import partial

from utils import from_cli
from .model import MachineModel, DeepPolicyMachineModel
from .deep_multi_rule import DeepMultiRule
from .rule import SchedulingRule
from .static import StaticModel as StaticMachineModel

key_to_class = {
    "static": StaticMachineModel,
    'deep_multi_rule': DeepMultiRule
}

from_cli = partial(from_cli, key_to_class=key_to_class)
