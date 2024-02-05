from abc import ABCMeta

from model.routing.static import routing_rules


class RoutingModel(routing_rules.RoutingRule, metaclass=ABCMeta):
    ...