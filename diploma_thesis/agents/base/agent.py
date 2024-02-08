from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic

import torch

from agents.utils import Phase, EvaluationPhase
from .encoder import Encoder as StateEncoder, Input, State
from .model import Model, Action, Result


class Agent(metaclass=ABCMeta):

    def __init__(self,
                 model: Model[Input, State, Action, Result],
                 state_encoder: StateEncoder[Input, State],
                 memory):
        self.state_encoder = state_encoder
        self.model = model
        self.memory = memory
        self.phase = EvaluationPhase()

    def update(self, phase: Phase):
        self.phase = phase

    @property
    @abstractmethod
    def is_trainable(self):
        pass

    @abstractmethod
    def train_step(self):
        pass

    def store(self,
              state: State,
              action: Action,
              next_state: State,
              reward: torch.FloatTensor):
        pass

    def schedule(self, parameters: Input) -> Model.Record:
        state = self.encode_state(parameters)

        return self.model(state, parameters)

    def encode_state(self, parameters: Input) -> State:
        return self.state_encoder.encode(parameters)
