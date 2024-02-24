from .routing_rule import RoutingRule
from .random import RandomRoutingRule
from .tt import TTRoutingRule
from .sq import SQRoutingRule
from .ea import EARoutingRule
from .ct import CTRoutingRule
from .et import ETRoutingRule
from .ut import UTRoutingRule
from .idle import IdleRoutingRule

ALL_ROUTING_RULES = {
    'sq': SQRoutingRule,
    'random': RandomRoutingRule,
    'tt': TTRoutingRule,
    'ea': EARoutingRule,
    'ct': CTRoutingRule,
    'et': ETRoutingRule,
    'ut': UTRoutingRule,
}
