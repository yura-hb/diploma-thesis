import environment

from typing import TypeVar, List
from agents import Phase, EvaluationPhase, Agent

MachineState = TypeVar('MachineState')


class WorkCenter(Agent):

    def __init__(self):
        self.phase = EvaluationPhase()

    @property
    def idx(self):
        return ''

    @property
    def is_trainable(self):
        return False

    def update(self, phase: Phase):
        self.phase = phase

    def train_step(self):
        pass

    def encode_state(self,
                     job: environment.Job,
                     work_center_idx: int,
                     machines: List[environment.Machine]) -> MachineState:
        pass

    def schedule(self, state: MachineState) -> environment.Job | environment.WaitInfo:
        pass

    def record(self, state: MachineState, action, next_state: MachineState, reward):
        pass

