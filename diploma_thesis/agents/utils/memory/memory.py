
from abc import ABCMeta, abstractmethod
from typing import TypeVar

import torch
from tensordict.prototype import tensorclass
from torchrl.data import TensorDictReplayBuffer

State = TypeVar('State')
Action = TypeVar('Action')
Configuration = TypeVar('Configuration')


@tensorclass
class Record:
    state: State
    action: Action
    next_state: State
    reward: torch.FloatTensor
    done: torch.BoolTensor


class NotReadyException(BaseException):
    pass


class Memory(metaclass=ABCMeta):

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.buffer: TensorDictReplayBuffer = self.__make_buffer__()

    def store(self, record: Record):
        self.buffer.extend(record)

    def sample(self, return_info: bool = False) -> Record:
        if len(self.buffer) < self.buffer._batch_size:
            raise NotReadyException()

        return self.buffer.sample(return_info=return_info)

    def sample_n(self, batch_size: int) -> Record:
        return self.buffer.sample(batch_size=batch_size)

    def update_priority(self, indices: torch.LongTensor, priorities: torch.FloatTensor):
        self.buffer.update_priority(indices, priorities)

    @abstractmethod
    def __make_buffer__(self) -> TensorDictReplayBuffer:
        pass

    def clear(self):
        self.buffer.empty()

    def __len__(self) -> int:
        return len(self.buffer)

    # TorchRL buffer isn't yet pickable. Hence, we recreate it from the configuration

    def __getstate__(self):
        return self.configuration

    def __setstate__(self, state):
        self.configuration = state
        self.buffer = self.__make_buffer__()
