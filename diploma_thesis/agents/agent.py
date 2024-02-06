

from .utils import Phase


class Agent:

    @property
    def is_trainable(self):
        return False

    def update(self, phase: Phase):
        self.phase = phase

    def train_step(self):
        pass