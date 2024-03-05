from agents.utils import TrainingPhase
from agents.utils.rl import RLTrainer
from utils import filter
from .agent import *
from .model import DeepPolicyModel


class RLAgent(Generic[Key], Agent[Key]):

    def __init__(self, model: DeepPolicyModel, state_encoder: StateEncoder, trainer: RLTrainer):
        super().__init__(model, state_encoder)

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
        self.trainer.store(sample)

    def loss_record(self):
        return self.trainer.loss_record()

    def clear_memory(self):
        self.trainer.clear()

    def schedule(self, key, parameters):
        result = super().schedule(key, parameters)

        if not self.trainer.is_configured:
            self.trainer.configure(self.model.policy)

        return result
