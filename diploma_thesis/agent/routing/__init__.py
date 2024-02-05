from typing import Dict

from .routing_model import RoutingModel
from agent.routing.static.static_routing_model import StaticRoutingModel

key_to_model = {
    'static': StaticRoutingModel,
}


def from_cli_arguments(configuration: Dict) -> RoutingModel:
    ...
