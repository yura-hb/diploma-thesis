from functools import partial

from utils import from_cli
from .model import WorkCenterModel, NNWorkCenterModel
from .rule import RoutingRule
from .static import StaticModel as StaticWorkCenterModel
from .multi_rule_linear import MultiRuleLinear

key_to_class = {
    "static": StaticWorkCenterModel,
    "multi_rule_linear": MultiRuleLinear
}


from_cli = partial(from_cli, key_to_class=key_to_class)