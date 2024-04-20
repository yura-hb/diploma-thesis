from agents.utils import TrainingPhase
from agents.utils.run_configuration import RunConfiguration
from agents.utils.rl import RLTrainer
from utils import filter
from .agent import *
from .model import DeepPolicyModel


class RLAgent(Generic[Key], Agent[Key]):

    def __init__(
            self, model: DeepPolicyModel, encoder: StateEncoder, trainer: RLTrainer, configuration: RunConfiguration
    ):
        super().__init__(model, encoder)

        self.configuration = configuration
        self.model: DeepPolicyModel = model
        self.trainer = trainer

        self.model.configure(configuration)

    def schedule(self, key, parameters):
        result = super().schedule(key, parameters)

        if not self.trainer.is_configured:
            self.trainer.configure(self.model.policy)

        return result

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

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),
            trainer=self.trainer.state_dict()
        )

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict['model'])
        self.trainer.load_state_dict(state_dict['trainer'])
