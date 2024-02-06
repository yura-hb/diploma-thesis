import environment

from typing import TypeVar, List
from agents import Phase

MachineState = TypeVar('MachineState')


class WorkCenter:

    def update(self, phase: Phase):
        pass

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

