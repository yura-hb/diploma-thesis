
from dataclasses import dataclass
from typing import Dict

from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer, PrioritizedReplayBuffer, ListStorage

from .memory import *


@dataclass
class Configuration(Configuration):
    alpha: float
    beta: float

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            alpha=parameters.get('alpha', 0.6),
            beta=parameters.get('beta', 0.4),
            size=parameters['size'],
            is_tensordict_storage=parameters.get('is_tensordict_storage', False),
            batch_size=parameters['batch_size'],
            prefetch=parameters.get('prefetch', 1)
        )


class PrioritizedReplayMemory(Memory[Configuration]):

    def __make_buffer__(self) -> PrioritizedReplayBuffer | TensorDictPrioritizedReplayBuffer:
        cls = None

        params = dict(
            batch_size=self.configuration.batch_size,
            prefetch=self.configuration.prefetch,
            alpha=self.configuration.alpha,
            beta=self.configuration.beta
        )

        return self.__make_result_buffer__(params, PrioritizedReplayBuffer, TensorDictPrioritizedReplayBuffer)

    @staticmethod
    def from_cli(parameters: Dict) -> 'PrioritizedReplayMemory':
        configuration = Configuration.from_cli(parameters)

        return PrioritizedReplayMemory(configuration)
