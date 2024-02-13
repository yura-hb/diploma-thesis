
from abc import ABCMeta, abstractmethod

# There are two types of rewards:
# 1. Reward can be assigned on the job completion using decomposition
# 2. Reward can be assigned on the operation completion using surrogate reward shaping
# Note that it is either the first method or the second one but not both


class RewardModel(metaclass=ABCMeta):

    @abstractmethod
    def prepare_context_before_operation(self):
        pass

    @abstractmethod
    def prepare_context_after_operation(self):
        pass

    @abstractmethod
    def reward(self):
        pass
