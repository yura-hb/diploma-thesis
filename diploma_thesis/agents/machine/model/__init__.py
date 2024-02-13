from .model import MachineModel
from .multi_rule_linear import MultiRuleLinear as MultiRuleLinearModel
from .rule import SchedulingRule
from .static import StaticModel as StaticMachineModel

key_to_class = {
    "static": StaticMachineModel,
    'multi_rule_linear': MultiRuleLinearModel
}


def from_cli(parameters) -> MachineModel:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters['parameters'])

