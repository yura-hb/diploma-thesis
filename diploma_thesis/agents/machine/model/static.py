
from .model import MachineModel
from agents.machine.state import PlainEncoder
from .rule import SchedulingRule, ALL_SCHEDULING_RULES
from typing import Dict


class StaticModel(MachineModel[PlainEncoder.State, None]):

    State = PlainEncoder.State

    def __init__(self, rule: SchedulingRule):
        self.rule = rule

    def __call__(self, state: State, parameters: MachineModel.Input) -> MachineModel.Record:
        return MachineModel.Record(
            result=self.rule(parameters.machine, parameters.now),
            state=state,
            action=None
        )

    @staticmethod
    def from_cli(parameters: Dict):
        rule = parameters['rule']
        rule = ALL_SCHEDULING_RULES[rule]

        return StaticModel(rule())
