from abc import ABCMeta

from agent.dispatch.scheduling import scheduling_rules


class SchedulingModel(scheduling_rules.SchedulingRule, metaclass=ABCMeta):
    ...
