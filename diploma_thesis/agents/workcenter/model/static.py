
import torch

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
            result=self.rule(job=parameters.job, work_center=parameters.work_center),
            state=state,
            action=None
        )

    def values(self, state: State) -> torch.FloatTensor:
        return torch.zeros(1, dtype=torch.float32)

    @staticmethod
    def from_cli(parameters: Dict):
        rule = parameters['rule']
        rule = ALL_ROUTING_RULES[rule]

        return StaticModel(rule())
