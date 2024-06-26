from typing import Dict

from agents.base.state import State
from .model import MachineModel
from .rule import SchedulingRule, ALL_SCHEDULING_RULES


class StaticModel(MachineModel[None]):

    def __init__(self, rule: SchedulingRule):
        super().__init__()
        self.rule = rule

    def __call__(self, state: State, parameters: MachineModel.Input) -> MachineModel.Record:
        return MachineModel.Record(
            result=self.rule(parameters.machine, parameters.now),
            record=None,
            batch_size=[]
        )

    @staticmethod
    def from_cli(parameters: Dict):
        rule = parameters['rule']
        rule = ALL_SCHEDULING_RULES[rule]

        return StaticModel(rule())
