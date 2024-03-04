
from dataclasses import dataclass

from torchrl.data import LazyMemmapStorage

from .memory import *
from .sampler import *


class ReplayMemory(Memory):

    @dataclass
    class Configuration:
        size: int
        batch_size: int
        prefetch: int
        sampler: Sampler

        @staticmethod
        def from_cli(parameters: Dict):
            return ReplayMemory.Configuration(
                size=parameters['size'],
                batch_size=parameters['batch_size'],
                prefetch=parameters.get('prefetch', 0),
                sampler=Sampler.from_cli(parameters['sampler']) if 'sampler' in parameters else None
            )

    def __make_buffer__(self) -> TensorDictReplayBuffer:
        storage = LazyMemmapStorage(max_size=self.configuration.size)
        sampler = self.configuration.sampler.make() if self.configuration.sampler else RandomSampler()

        return TensorDictReplayBuffer(storage=storage,
                                      batch_size=self.configuration.batch_size,
                                      sampler=sampler,
                                      prefetch=self.configuration.prefetch)

    @staticmethod
    def from_cli(parameters: Dict) -> 'ReplayMemory':
        configuration = ReplayMemory.Configuration.from_cli(parameters)

        return ReplayMemory(configuration)

    @classmethod
    def load_from_parameters(cls, parameters):
        return cls(parameters)
