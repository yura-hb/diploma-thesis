from functools import partial

from utils import from_cli
from .model import MachineModel, DeepPolicyMachineModel
from .deep_rule import DeepRule
from .deep_multi_rule import DeepMultiRule
from .deep_mr import DeepMR
from .rule import SchedulingRule
from .static import StaticModel as StaticMachineModel

key_to_class = {
    "static": StaticMachineModel,
    'deep_rule': DeepRule,
    'deep_multi_rule': DeepMultiRule,
    'deep_mr': DeepMR
}

from_cli = partial(from_cli, key_to_class=key_to_class)
