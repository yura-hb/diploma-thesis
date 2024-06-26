
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, List

from agents.utils import Phase, EvaluationPhase, PhaseUpdatable
from agents.utils.memory import Record
from agents.utils.action import ActionSelector
from environment import ShopFloor
from utils import Loggable
from .encoder import Encoder as StateEncoder, Input, State
from .model import Model, Action, Result

Key = TypeVar('Key')


@dataclass
class TrainingSample:
    episode_id: int
    records: List[Record]


@dataclass
class Slice(TrainingSample):
    pass


@dataclass
class Trajectory(TrainingSample):
    pass


class Agent(Generic[Key], Loggable, PhaseUpdatable, metaclass=ABCMeta):

    def __init__(self, model: Model[Input, Action, Result], state_encoder: StateEncoder[Input]):
        self.state_encoder = state_encoder
        self.model = model
        self.phase = EvaluationPhase()

        super().__init__()

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

    def with_action_selector(self, action_selector: ActionSelector):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict: dict):
        pass

    def schedule(self, key: Key, parameters: Input) -> Model.Record:
        state = self.encode_state(parameters)

        return self.model(state, parameters)

    def encode_state(self, parameters: Input) -> State:
        return self.state_encoder.encode(parameters)


