
import logging
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List

from agents.utils import Phase, EvaluationPhase, PhaseUpdatable
from agents.utils.memory import Record
from environment import ShopFloor
from utils import Loggable
from .encoder import Encoder as StateEncoder, Input, State
from .model import Model, Action, Result
from dataclasses import dataclass

Key = TypeVar('Key')


@dataclass
class TrainingSample:
    episode_id: int


@dataclass
class Slice(TrainingSample):
    records: List[Record]


@dataclass
class Trajectory(TrainingSample):
    records: List[Record]


class Agent(Generic[Key], Loggable, PhaseUpdatable, metaclass=ABCMeta):

    def __init__(self,
                 model: Model[Input, State, Action, Result],
                 state_encoder: StateEncoder[Input, State]):
        self.state_encoder = state_encoder
        self.model = model
        self.phase = EvaluationPhase()
        super().__init__()
        self.__post_init__()

    def __post_init__(self):
        pass

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        for module in [self.model, self.state_encoder]:
            if isinstance(module, Loggable):
                module.with_logger(logger)

        return self

    def update(self, phase: Phase):
        self.phase = phase

        for module in [self.model, self.state_encoder]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def setup(self, shop_floor: ShopFloor):
        pass

    @property
    @abstractmethod
    def is_trainable(self):
        return self.phase != EvaluationPhase()

    @property
    @abstractmethod
    def is_distributed(self):
        return False

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def store(self, key: Key, sample: TrainingSample):
        pass

    def schedule(self, key: Key, parameters: Input) -> Model.Record:
        state = self.encode_state(parameters)

        return self.model(state, parameters)

    def encode_state(self, parameters: Input) -> State:
        return self.state_encoder.encode(parameters)


