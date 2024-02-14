
from .memory import Record, Memory
from typing import Dict
from .replay_memory import ReplayMemory

key_to_cls = {
    'replay': ReplayMemory
}


def from_cli(parameters: Dict) -> Memory:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
