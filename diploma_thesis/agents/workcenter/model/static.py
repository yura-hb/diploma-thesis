from typing import Dict

from agents.machine.state import PlainEncoder
from .model import WorkCenterModel
from .rule import RoutingRule, ALL_ROUTING_RULES


class StaticModel(WorkCenterModel[PlainEncoder.State, None]):

    State = PlainEncoder.State

    def __init__(self, rule: RoutingRule):
        self.rule = rule
        super().__init__()

    def __call__(self, state: State, parameters: WorkCenterModel.Input) -> WorkCenterModel.Record:
        return WorkCenterModel.Record(
            result=self.rule(parameters.job, parameters.work_center_idx, parameters.machines),
            state=state,
            action=None
        )

    @staticmethod
    def from_cli(parameters: Dict):
        rule = parameters['rule']
        rule = ALL_ROUTING_RULES[rule]

        return StaticModel(rule())
