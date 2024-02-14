
from .memory import *
from typing import Dict
from torchrl.data import LazyMemmapStorage


class ReplayMemory(Memory):

    def __init__(self,
                 size: int,
                 batch_size: int,
                 prefetch: int):
        storage = LazyMemmapStorage(max_size=size)
        buffer = TensorDictReplayBuffer(storage=storage, batch_size=batch_size, prefetch=prefetch)

        super().__init__(buffer=buffer)

    @staticmethod
    def from_cli(parameters: Dict) -> 'ReplayMemory':
        return ReplayMemory(size=parameters['size'],
                            batch_size=parameters['batch_size'],
                            prefetch=parameters.get('prefetch', 1))
