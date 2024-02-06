import environment

from typing import TypeVar
from agents import Phase, EvaluationPhase

MachineState = TypeVar('MachineState')


class Machine:

    def __init__(self, state_encoder, model, memory):
        self.state_encoder = state_encoder
        self.model = model
        self.memory = memory
        self.phase = EvaluationPhase()

    @property
    def is_trainable(self):
        return False

    def update(self, phase: Phase):
        self.phase = phase

    def train_step(self):
        pass

    def encode_state(self, machine: environment.Machine) -> MachineState:
        pass

    def schedule(self, state: MachineState) -> environment.Job | environment.WaitInfo:
        pass

    def record(self, state: MachineState, action, next_state: MachineState, reward):
        pass

