from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TypeVar, List, Generic, Dict

import torch
from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torchrl.data import LazyMemmapStorage, ListStorage
from torchrl.data import ReplayBuffer

State = TypeVar('State')
Action = TypeVar('Action')
_Configuration = TypeVar('_Configuration')


@dataclass
class Configuration:
    size: int
    is_tensordict_storage: bool
    batch_size: int
    prefetch: int


@tensorclass
class Record:
    POLICY_KEY = "policy"
    VALUE_KEY = "value"
    REWARD_KEY = "reward"
    RETURN_KEY = "return"
    ACTION_KEY = "actions"
    ADVANTAGE_KEY = "advantage"

    state: State
    action: Action
    next_state: State
    reward: torch.FloatTensor
    done: torch.BoolTensor
    info: TensorDict = field(default_factory=lambda: TensorDict({}, batch_size=[]))

    @property
    def is_filled(self):
        return (self.state is not None and
                self.action is not None and
                self.next_state is not None and
                self.reward is not None and
                self.done is not None)


class NotReadyException(BaseException):
    pass

stores = 0

class Memory(Generic[_Configuration], metaclass=ABCMeta):

    def __init__(self, configuration: _Configuration):
        self.configuration = configuration
        self.buffer: ReplayBuffer = self.__make_buffer__()

    def store(self, records: List[Record] | List[List[Record]]):
        if self.configuration.is_tensordict_storage:
            records = torch.cat(records, dim=0)

            self.buffer.extend(records)
        else:
            global stores

            stores += 1

            print(f'Stores {stores} {len(records)}')

            self.buffer.extend(records)

    def sample(self, return_info: bool = False) -> List[Record]:
        if len(self.buffer) < self.configuration.batch_size:
            raise NotReadyException()

        result, index = self.buffer.sample(return_info=True)
        result = self.__clone__(result)

        if self.configuration.is_tensordict_storage:
            if not isinstance(result, list):
                result = [result]

            return list(record.unbind(dim=0) for record in result), index

        return result, index

    def sample_n(self, batch_size: int) -> Record:
        return self.buffer.sample(batch_size=batch_size)

    def update_priority(self, indices: torch.LongTensor, priorities: torch.Tensor):
        self.buffer.update_priority(indices, priorities)

    @abstractmethod
    def __make_buffer__(self) -> ReplayBuffer:
        pass

    def clear(self):
        self.buffer.empty()

    def __len__(self) -> int:
        return len(self.buffer)

    def __make_result_buffer__(self, params: Dict, regular_cls, tensordict_cls):
        if self.configuration.is_tensordict_storage:
            params['storage'] = LazyMemmapStorage(max_size=self.configuration.size)
            cls = tensordict_cls
        else:
            params['storage'] = ListStorage(max_size=self.configuration.size)
            params['collate_fn'] = lambda x: x
            cls = regular_cls

        return cls(**params)

    # TorchRL buffer isn't yet pickable. Hence, we recreate it from the configuration

    def __getstate__(self):
        return self.configuration

    def __setstate__(self, state):
        self.configuration = state
        self.buffer = self.__make_buffer__()

    @classmethod
    def __clone__(cls, batch):
        if isinstance(batch, Record):
            return batch.clone()

        if isinstance(batch, list):
            return [cls.__clone__(element) for element in batch]
