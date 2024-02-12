
import logging
from abc import ABCMeta, abstractmethod

import torch

from agents.utils import Phase, EvaluationPhase, Loggable
from .encoder import Encoder as StateEncoder, Input, State
from .model import Model, Action, Result


class Agent(Loggable, metaclass=ABCMeta):

    def __init__(self,
                 model: Model[Input, State, Action, Result],
                 state_encoder: StateEncoder[Input, State],
                 memory):
        self.state_encoder = state_encoder
        self.model = model
        self.memory = memory
        self.phase = EvaluationPhase()
        super().__init__()

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        for module in [self.memory, self.model, self.state_encoder]:
            if isinstance(module, Loggable):
                module.with_logger(logger)

        return self

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
