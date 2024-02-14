
import logging
from abc import ABCMeta, abstractmethod

from agents.utils import Phase, EvaluationPhase, Loggable, PhaseUpdatable
from agents.utils.memory import Memory, Record
from .encoder import Encoder as StateEncoder, Input, State
from .model import Model, Action, Result
from typing import TypeVar, Generic

Key = TypeVar('Key')


class Agent(Generic[Key], Loggable, PhaseUpdatable, metaclass=ABCMeta):

    def __init__(self,
                 model: Model[Input, State, Action, Result],
                 state_encoder: StateEncoder[Input, State],
                 memory: Memory):
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

        for module in [self.memory, self.model, self.state_encoder]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    @property
    @abstractmethod
    def is_trainable(self):
        return self.phase != EvaluationPhase()

    @abstractmethod
    def train_step(self):
        pass

    def store(self, key: Key, record: Record):
        pass

    def schedule(self, parameters: Input) -> Model.Record:
        state = self.encode_state(parameters)

        return self.model(state, parameters)

    def encode_state(self, parameters: Input) -> State:
        return self.state_encoder.encode(parameters)
