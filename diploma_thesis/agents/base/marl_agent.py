import copy
from typing import Dict

from agents.utils import TrainingPhase
from agents.utils.rl import RLTrainer
from utils import filter
from .agent import *
from .model import NNModel


class MARLAgent(Generic[Key], Agent[Key]):

    def __init__(self, model: NNModel, state_encoder: StateEncoder, trainer: RLTrainer, is_model_distributed: bool):
        super().__init__(model, state_encoder)

        self.trainer: RLTrainer | Dict[Key, RLTrainer] = trainer
        self.is_model_distributed = is_model_distributed
        self.is_configured = False
        self.keys = None

    @property
    def is_trainable(self):
        return True

    @abstractmethod
    def iterate_keys(self, shop_floor: ShopFloor):
        pass

    def setup(self, shop_floor: ShopFloor):
        if self.is_configured:
            is_key_set_equal = set(self.keys) == set(self.iterate_keys(shop_floor))
            is_evaluating_with_centralized_model = self.phase == EvaluationPhase() and not self.is_model_distributed

            assert is_key_set_equal or is_evaluating_with_centralized_model, \
                ("Multi-Agent model should be configured for the same shop floor architecture "
                 "or have centralized action network")
            return

        self.is_configured = True
        self.keys = list(self.iterate_keys(shop_floor))

        base_model = self.model
        base_trainer = self.trainer

        self.model = dict() if self.is_model_distributed else self.model
        self.trainer = dict()

        for key in self.keys:
            if self.is_model_distributed:
                self.model[key] = copy.deepcopy(base_model)

            self.trainer[key] = copy.deepcopy(base_trainer)

    @filter(lambda self: self.phase == TrainingPhase())
    def train_step(self):
        for key in self.keys:
            self.trainer[key].train_step(self.__model_for_key__(key))

    @filter(lambda self, *args: self.phase == TrainingPhase())
    def store(self, key: Key, record: Record):
        self.trainer[key].store(record)

    def loss_record(self):
        result = [self.trainer[key].loss_record() for key in self.keys]

        return result

    def clear_memory(self):
        for key in self.keys:
            self.trainer[key].clear()

    def schedule(self, key: Key, parameters):
        state = self.encode_state(parameters)

        result = self.__model_for_key__(key)(state, parameters)

        if not self.trainer[key].is_configured:
            self.trainer[key].configure(self.model)

        return result

    def __model_for_key__(self, key: Key):
        if self.is_model_distributed:
            return self.model[key]

        return self.model
