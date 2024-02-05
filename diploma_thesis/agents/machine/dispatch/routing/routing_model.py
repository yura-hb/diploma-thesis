from abc import ABCMeta

from agents.machine.dispatch.routing.static import routing_rules


class RoutingModel(routing_rules.RoutingRule, metaclass=ABCMeta):
    ...
