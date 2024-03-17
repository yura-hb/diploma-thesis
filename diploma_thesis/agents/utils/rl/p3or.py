from agents.utils.memory import NotReadyException
from .utils.ppo_mixin import *


@dataclass
class Configuration(PPOConfiguration):
    trpo_penalty: bool

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            **PPOConfiguration.base_parameters_from_cli(parameters),
            trpo_penalty=parameters.get('trpo_penalty', 0.1)
        )


class P3OR(PPOMixin):

    def __init__(self, configuration: Configuration, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.configuration = configuration

    def __configure__(self, model: Policy, configuration: RunConfiguration):
        pass

    def __train__(self, model: Policy):
        try:
            batch, generator = self.storage.sample_minibatches(update_returns=self.configuration.update_advantages,
                                                               device=self.run_configuration.device,
                                                               n=self.configuration.epochs,
                                                               sample_ratio=self.configuration.sample_ratio)

            for minibatch in generator:
                self.__step__(minibatch, model, self.configuration)

            # Auxiliary step
            self.__auxiliary_step__(batch)
        except NotReadyException:
            return

    def __auxiliary_step__(self, batch: Batch):
        pass

    @classmethod
    def from_cli(cls,
                 parameters: Dict,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator):
        schedule = TrainSchedule.from_cli(parameters)
        configuration = Configuration.from_cli(parameters)

        return cls(configuration=configuration,
                   memory=memory,
                   optimizer=optimizer,
                   loss=loss,
                   return_estimator=return_estimator,
                   train_schedule=schedule)
