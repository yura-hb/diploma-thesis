
import torch

from abc import ABCMeta
from typing import TypeVar
from dataclasses import dataclass
from simulator import Simulation

State = TypeVar('State')
Action = TypeVar('Action')


@dataclass
class Record:
    state: State
    action: Action
    next_state: State
    reward: torch.FloatTensor
    done: bool


class RewardModel(metaclass=ABCMeta):

    def __init__(self):
        pass

    def connect(self, simulator: 'Simulator'):
        pass

    def did_record_state(self):
        pass
