
from dataclasses import dataclass
from typing import Dict

from agents.utils.memory import NotReadyException
from .utils.ppo_mixin import *


@dataclass
class Configuration(PPOConfiguration):

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(**PPOConfiguration.base_parameters_from_cli(parameters))


class PPO(PPOMixin):

    def __init__(self, configuration: Configuration, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.configuration = configuration

    @property
    def is_episodic(self):
        return True

    def __train__(self, model: Policy):
        try:
            _, generator = self.storage.sample_minibatches(update_returns=self.configuration.update_advantages,
                                                           device=self.run_configuration.device,
                                                           n=self.configuration.epochs,
                                                           sample_ratio=self.configuration.sample_ratio)

            for minibatch in generator:
                self.__step__(minibatch, model, self.configuration)
        except NotReadyException:
            return

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

