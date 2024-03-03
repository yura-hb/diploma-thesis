from functools import partial

from utils import from_cli
from .rule import RoutingRule
from .model import DeepPolicyWorkCenterModel, WorkCenterModel
from .static import StaticModel as StaticWorkCenterModel
from .deep_multi_rule import DeepMultiRule

key_to_class = {
    "static": StaticWorkCenterModel,
    "deep_multi_rule": DeepMultiRule
}


from_cli = partial(from_cli, key_to_class=key_to_class)