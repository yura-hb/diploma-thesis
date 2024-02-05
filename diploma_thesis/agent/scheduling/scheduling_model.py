from abc import ABCMeta

from agent.scheduling import scheduling_rules


class SchedulingModel(scheduling_rules.SchedulingRule, metaclass=ABCMeta):
    ...
