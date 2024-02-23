
from .memory import *
from typing import Dict
from torchrl.data import LazyMemmapStorage

from dataclasses import dataclass


class ReplayMemory(Memory):

    @dataclass
    class Configuration:
        size: int
        batch_size: int
        prefetch: int = 1

        @staticmethod
        def from_cli(parameters: Dict):
            return ReplayMemory.Configuration(
                size=parameters['size'],
                batch_size=parameters['batch_size'],
                prefetch=parameters.get('prefetch', 1)
            )

    def __make_buffer__(self) -> TensorDictReplayBuffer:
        storage = LazyMemmapStorage(max_size=self.configuration.size)

        return TensorDictReplayBuffer(storage=storage,
                                      batch_size=self.configuration.batch_size,
                                      prefetch=self.configuration.prefetch)

    @staticmethod
    def from_cli(parameters: Dict) -> 'ReplayMemory':
        configuration = ReplayMemory.Configuration.from_cli(parameters)

        return ReplayMemory(configuration)

    @classmethod
    def load_from_parameters(cls, parameters):
        return cls(parameters)
