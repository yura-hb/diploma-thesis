import torch

from agents.utils import TrainingPhase
from agents.utils.rl import RLTrainer
from utils import filter
from .agent import *
from .model import DeepPolicyModel

from dataclasses import dataclass


@dataclass
class Configuration:
    compile: bool = False

    @staticmethod
    def from_cli(parameters):
        return Configuration(
            compile=parameters.get('compile', False)
        )


class RLAgent(Generic[Key], Agent[Key]):

    def __init__(self,
                 model: DeepPolicyModel,
                 state_encoder: StateEncoder,
                 trainer: RLTrainer,
                 configuration: Configuration):
        super().__init__(model, state_encoder)

        self.is_compiled = False

        self.configuration = configuration
        self.model: DeepPolicyModel = model
        self.trainer = trainer

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        for module in [self.model, self.state_encoder, self.trainer]:
            if isinstance(module, Loggable):
                module.with_logger(logger)

        return self

    @property
    def is_trainable(self):
        return True

    @property
    def is_distributed(self):
        return False

    @filter(lambda self: self.phase == TrainingPhase())
    def train_step(self):
        self.trainer.train_step(self.model.policy)

    @filter(lambda self, *args: self.phase != EvaluationPhase())
    def store(self, key: Key, sample: TrainingSample):
        self.trainer.store(sample, self.model.policy)

    def loss_record(self):
        return self.trainer.loss_record()

    def clear_memory(self):
        self.trainer.clear()

    def schedule(self, key, parameters):
        result = super().schedule(key, parameters)

        if not self.trainer.is_configured:
            self.trainer.configure(self.model.policy)

        if not self.is_compiled:
            self.compile()

        return result

    def __setstate__(self, state):
        self.__dict__ = state

        self.compile()

    def compile(self):
        if not self.configuration.compile:
            self.is_compiled = True
            return

        if self.is_compiled:
            return

        self.trainer.compile()
        self.model.compile()

        self.is_compiled = True
