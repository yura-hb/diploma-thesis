
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
        super().__init__(configuration, *args, **kwargs)

        self.configuration: Configuration = configuration

    @property
    def is_episodic(self):
        return True

    def __train__(self, model: Policy):
        try:
            _, generator, info = self.storage.sample_minibatches(device=self.device,
                                                                 n=self.configuration.epochs,
                                                                 sample_count=self.configuration.sample_count)

            for minibatch in generator:
                self.__step__(minibatch, model)

            self.__increase_memory_priority__(info)
        except NotReadyException:
            return

    @classmethod
    def from_cli(cls, parameters: Dict, **kwargs):
        schedule = TrainSchedule.from_cli(parameters)
        configuration = Configuration.from_cli(parameters)

        return cls(configuration=configuration, train_schedule=schedule, **kwargs)

