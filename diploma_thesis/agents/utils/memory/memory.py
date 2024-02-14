
from abc import ABCMeta
from typing import TypeVar

import torch
from tensordict.prototype import tensorclass
from torchrl.data import TensorDictReplayBuffer

State = TypeVar('State')
Action = TypeVar('Action')


@tensorclass
class Record:
    state: State
    action: Action
    next_state: State
    reward: torch.FloatTensor
    done: torch.BoolTensor


class Memory(metaclass=ABCMeta):

    def __init__(self, buffer: TensorDictReplayBuffer):
        self.buffer = buffer

    def store(self, record: Record):
        self.buffer.extend(record)

    def sample(self) -> Record:
        return self.buffer.sample()

    def sample_n(self, batch_size: int) -> Record:
        return self.buffer.sample(batch_size=batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
