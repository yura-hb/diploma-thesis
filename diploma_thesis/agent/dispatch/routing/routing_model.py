from abc import ABCMeta

from agent.dispatch.routing.static import routing_rules


class RoutingModel(routing_rules.RoutingRule, metaclass=ABCMeta):
    ...
