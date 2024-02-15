
from .work_center import *
from utils import load
from typing import Dict


class PersistedWorkCenter(WorkCenter):

    @property
    def is_trainable(self):
        return False

    def train_step(self):
        pass

    @staticmethod
    def from_cli(parameters: Dict):
        return load(parameters['path'])
