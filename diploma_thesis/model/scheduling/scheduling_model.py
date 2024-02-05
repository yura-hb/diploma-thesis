from abc import ABCMeta

from model.scheduling import scheduling_rules


class SchedulingModel(scheduling_rules.SchedulingRule, metaclass=ABCMeta):
    ...
