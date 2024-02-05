
from .routing_rule import RoutingRule, WorkCenterState

from .random import RandomRoutingRule
from .tt import TTRoutingRule
from .sq import SQRoutingRule
from .ea import EARoutingRule
from .ct import CTRoutingRule
from .et import ETRoutingRule
from .ut import UTRoutingRule

ALL_ROUTING_RULES = {
    'random': RandomRoutingRule,
    'tt': TTRoutingRule,
    'sq': SQRoutingRule,
    'ea': EARoutingRule,
    'ct': CTRoutingRule,
    'et': ETRoutingRule,
    'ut': UTRoutingRule,
}
