import environment
from environment import Job
from model.scheduling.scheduling_rules import ALL_SCHEDULING_RULES
from environment.scheduling_rule import WaitInfo
from model.scheduling.scheduling_model import SchedulingModel


class StaticSchedulingModel(SchedulingModel):

    def __init__(self, rule: str):
        self.rule = ALL_SCHEDULING_RULES[rule]()

        super().__init__()

    def __call__(self, machine: environment.Machine, now: float) -> Job | WaitInfo:
        return self.rule(machine, now)
