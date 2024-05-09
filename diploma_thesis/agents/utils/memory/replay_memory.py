from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement

from .memory import *
from .memory import Configuration as MemoryConfiguration
from .sampler import *


@dataclass
class Configuration(MemoryConfiguration):
    sampler: Sampler

    @classmethod
    def from_cli(cls, parameters: Dict):
        return cls(
            size=parameters['size'],
            batch_size=parameters['batch_size'],
            prefetch=parameters.get('prefetch', 0),
            is_tensordict_storage=parameters.get('is_tensordict_storage', False),
            sampler=Sampler.from_cli(parameters['sampler']) if 'sampler' in parameters else None
        )


class ReplayMemory(Memory[Configuration]):

    def __make_buffer__(self) -> ReplayBuffer | TensorDictReplayBuffer:
        sampler = self.configuration.sampler.make() if self.configuration.sampler else SamplerWithoutReplacement()
        cls = None

        params = dict(
            batch_size=self.configuration.batch_size,
            prefetch=self.configuration.prefetch,
            sampler=sampler
        )

        return self.__make_result_buffer__(params, ReplayBuffer, TensorDictReplayBuffer)

    @staticmethod
    def from_cli(parameters: Dict) -> 'ReplayMemory':
        configuration = Configuration.from_cli(parameters)

        return ReplayMemory(configuration)

    @classmethod
    def load_from_parameters(cls, parameters):
        return cls(parameters)
