from functools import partial

from utils import from_cli
from .model import MachineModel, NNMachineModel
from .multi_rule_linear import MultiRuleLinear as MultiRuleLinearModel
from .rule import SchedulingRule
from .static import StaticModel as StaticMachineModel

key_to_class = {
    "static": StaticMachineModel,
    'multi_rule_linear': MultiRuleLinearModel
}

from_cli = partial(from_cli, key_to_class=key_to_class)
