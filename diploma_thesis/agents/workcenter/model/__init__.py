from functools import partial

from utils import from_cli
from .model import WorkCenterModel
from .rule import RoutingRule
from .static import StaticModel as StaticWorkCenterModel

key_to_class = {
    "static": StaticWorkCenterModel
}


from_cli = partial(from_cli, key_to_class=key_to_class)