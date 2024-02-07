

from .machine import Machine
from .model import StaticMachineModel, SchedulingRule
from .state import PlainEncoder


class StaticMachine(Machine[StaticMachineModel, PlainEncoder]):

    def __init__(self, rule: SchedulingRule):
        super().__init__(model=StaticMachineModel(rule), state_encoder=PlainEncoder(), memory=None)

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass
