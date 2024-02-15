from .machine import *

from typing import Dict
from utils import load


class PersistedMachine(Machine):

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        return load(parameters['path'])
