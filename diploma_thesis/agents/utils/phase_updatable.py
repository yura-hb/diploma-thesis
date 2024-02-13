
from .phase import Phase


class PhaseUpdatable:

    def __init__(self):
        self.phase = None

    def update(self, phase: Phase):
        self.phase = phase
