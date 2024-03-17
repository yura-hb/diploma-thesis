from typing import Dict

from agents.base.state import State
from .model import WorkCenterModel
from .rule import RoutingRule, ALL_ROUTING_RULES


class StaticModel(WorkCenterModel[None]):

    def __init__(self, rule: RoutingRule):
        self.rule = rule
        super().__init__()

    def __call__(self, state: State, parameters: WorkCenterModel.Input) -> WorkCenterModel.Record:
        return WorkCenterModel.Record(
            result=self.rule(job=parameters.job, work_center=parameters.work_center),
            record=None,
            batch_size=[]
        )

    @staticmethod
    def from_cli(parameters: Dict):
        rule = parameters['rule']
        rule = ALL_ROUTING_RULES[rule]

        return StaticModel(rule())
