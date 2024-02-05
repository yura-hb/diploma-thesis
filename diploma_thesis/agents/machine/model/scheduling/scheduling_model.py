from abc import ABCMeta

from agents.machine.model.scheduling import scheduling_rules


class SchedulingModel(scheduling_rules.SchedulingRule, metaclass=ABCMeta):
    ...
