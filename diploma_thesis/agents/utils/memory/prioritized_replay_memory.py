
from .memory import *
from typing import Dict
from torchrl.data import ListStorage, TensorDictPrioritizedReplayBuffer


class PrioritizedReplayMemory(Memory):

    def __init__(self,
                 alpha: float,
                 beta: float,
                 size: int,
                 batch_size: int,
                 prefetch: int):
        storage = ListStorage(max_size=size)
        buffer = TensorDictPrioritizedReplayBuffer(
            alpha=alpha,
            beta=beta,
            batch_size=batch_size,
            prefetch=prefetch,
            storage=storage
        )

        super().__init__(buffer=buffer)

    @staticmethod
    def from_cli(parameters: Dict) -> 'PrioritizedReplayMemory':
        return PrioritizedReplayMemory(
            alpha=parameters.get('alpha', 0.6),
            beta=parameters.get('beta', 0.4),
            size=parameters['size'],
            batch_size=parameters['batch_size'],
            prefetch=parameters.get('prefetch', 1)
        )