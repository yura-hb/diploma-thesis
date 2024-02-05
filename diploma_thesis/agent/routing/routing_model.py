from abc import ABCMeta

from agent.routing.static import routing_rules


class RoutingModel(routing_rules.RoutingRule, metaclass=ABCMeta):
    ...
