from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic

import torch

from agents.utils import Phase, EvaluationPhase
from .encoder import Encoder as STEncoder
from .model import Model as Brain

StateEncoder = TypeVar('StateEncoder', bound=STEncoder)
Model = TypeVar('Model', bound=Brain)


class Agent(Generic[StateEncoder, Model], metaclass=ABCMeta):

    def __init__(self,
                 state_encoder: StateEncoder.State,
                 model: Model,
                 memory):
        self.state_encoder = state_encoder
        self.model = model
        self.memory = memory
        self.phase = EvaluationPhase()

        # TODO: Check if it works!!!

        assert isinstance(self.state_encoder.Input, self.model.Input), \
            f"State encoder input type {self.state_encoder.Input} does not match model input type {self.model.Input}"
        assert isinstance(self.state_encoder.State, self.model.State), \
            f"State encoder state type {self.state_encoder.State} does not match model state type {self.model.State}"

    def update(self, phase: Phase):
        self.phase = phase

    @abstractmethod
    @property
    def is_trainable(self):
        pass

    @abstractmethod
    def train_step(self):
        pass

    def store(self,
              state: StateEncoder.State,
              action: Model.Action,
              next_state: StateEncoder.State,
              reward: torch.FloatTensor):
        pass

    def schedule(self, state: StateEncoder.Input) -> Model.Record:
        state = self.encode_state(state)

        return self.model(state, state)

    def encode_state(self, state: StateEncoder.Input) -> StateEncoder.State:
        return self.state_encoder.encode(state)


