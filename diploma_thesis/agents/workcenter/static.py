

from .work_center import WorkCenter
from .model import StaticWorkCenterModel, RoutingRule
from .state import PlainEncoder


class StaticMachine(WorkCenter[StaticWorkCenterModel, PlainEncoder]):

    def __init__(self, rule: RoutingRule):
        super().__init__(model=StaticWorkCenterModel(rule), state_encoder=PlainEncoder(), memory=None)

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass
