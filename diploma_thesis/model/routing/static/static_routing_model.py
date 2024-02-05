
import environment

from model.routing.routing_model import RoutingModel
from model.routing.static.routing_rules import ALL_ROUTING_RULES, WorkCenterState


class StaticRoutingModel(RoutingModel):

    def __init__(self, rule: str):
        self.rule = ALL_ROUTING_RULES[rule]()

        super().__init__()

    def __call__(self, job: environment.Job, state: WorkCenterState) -> 'environment.Machine':
        return self.rule.select_machine(job, state)
