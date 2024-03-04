
from dataclasses import dataclass
from typing import Dict

from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer

from .memory import *


class PrioritizedReplayMemory(Memory):
    @dataclass
    class Configuration:
        alpha: float
        beta: float
        size: int
        batch_size: int
        prefetch: int = 1

        @staticmethod
        def from_cli(parameters: Dict):
            return PrioritizedReplayMemory.Configuration(
                alpha=parameters.get('alpha', 0.6),
                beta=parameters.get('beta', 0.4),
                size=parameters['size'],
                batch_size=parameters['batch_size'],
                prefetch=parameters.get('prefetch', 1)
            )

    def __make_buffer__(self) -> TensorDictPrioritizedReplayBuffer:
        storage = LazyMemmapStorage(max_size=self.configuration.size)

        return TensorDictPrioritizedReplayBuffer(
            storage=storage,
            batch_size=self.configuration.batch_size,
            prefetch=self.configuration.prefetch,
            alpha=self.configuration.alpha,
            beta=self.configuration.beta
        )

    @staticmethod
    def from_cli(parameters: Dict) -> 'PrioritizedReplayMemory':
        configuration = PrioritizedReplayMemory.Configuration.from_cli(parameters)

        return PrioritizedReplayMemory(configuration)
